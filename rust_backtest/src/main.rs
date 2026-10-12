use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{self, BufRead};
use std::path::Path;

#[derive(Clone, Debug)]
struct Row {
    timestamp: String,
    symbol: String,
    prediction: f64,
    target: f64,
    /// Raw (not market-demeaned) log return of the holding bar, if exported.
    raw_target: f64,
    quote_volume: f64,
    funding_rate: f64,
    /// Settlement rate paid during this row's holding window, if exported.
    funding_event_rate: f64,
}

#[derive(Clone, Copy, Debug)]
struct Columns {
    has_raw_target: bool,
    has_funding_events: bool,
}

#[derive(Clone, Copy, Debug)]
struct Config {
    long_percent: f64,
    short_percent: f64,
    signal_interval: usize,
    hold_bars: usize,
    gross_leverage: f64,
    initial_cash: f64,
    include_costs: bool,
    include_funding: bool,
    taker_fee_bps: f64,
    slippage_base_bps: f64,
    slippage_impact_coef: f64,
    max_slippage_bps: f64,
}

fn usage() {
    eprintln!(
        "Usage: crypto-factor-backtest --input predictions.csv --output metrics.json \\
         [--layers-output layers.csv] [--timeseries-output timeseries.csv] \\
         [--long-percent 10] [--short-percent 10] \\
         [--signal-interval-bars 1] [--signal-hold-bars 1] [--gross-leverage 1] \\
         [--include-costs] [--include-funding]"
    );
}

fn argument(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|pair| pair[0] == name).map(|pair| pair[1].clone())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|value| value == name)
}

fn parse_f64(args: &[String], name: &str, default: f64) -> f64 {
    argument(args, name).and_then(|v| v.parse::<f64>().ok()).unwrap_or(default)
}

fn parse_usize(args: &[String], name: &str, default: usize) -> usize {
    argument(args, name).and_then(|v| v.parse::<usize>().ok()).unwrap_or(default)
}

fn parse_csv_line(line: &str) -> Vec<String> {
    // The export format intentionally uses simple, comma-free fields.  This
    // keeps the standalone core dependency-free; Parquet/Arrow adapters can
    // be added without changing the evaluator itself.
    line.trim_end_matches(|c| c == '\r' || c == '\n')
        .split(',').map(str::to_owned).collect()
}

fn read_predictions(path: &str) -> io::Result<(String, Columns, BTreeMap<String, Vec<Row>>)> {
    let file = fs::File::open(path)?;
    let mut lines = io::BufReader::new(file).lines();
    let header = parse_csv_line(&lines.next().ok_or_else(|| io::Error::new(
        io::ErrorKind::InvalidData, "empty predictions CSV"))??);
    let index = |name: &str| header.iter().position(|v| v == name).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, format!("missing CSV column: {name}"))
    });
    let ti = index("timestamp")?;
    let si = index("symbol")?;
    let pi = index("prediction")?;
    let yi = index("target")?;
    let qi = index("quote_volume").ok();
    let fi = index("funding_rate").ok();
    let ri = index("raw_target").ok();
    let ei = index("funding_event_rate").ok();
    let columns = Columns { has_raw_target: ri.is_some(), has_funding_events: ei.is_some() };
    let factor_id = Path::new(path)
        .file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or("factor")
        .to_string();
    let mut groups: BTreeMap<String, Vec<Row>> = BTreeMap::new();
    for line in lines {
        let fields = parse_csv_line(&line?);
        let get = |i: usize| fields.get(i).map(String::as_str).unwrap_or("");
        let prediction = get(pi).parse::<f64>().unwrap_or(f64::NAN);
        let target = get(yi).parse::<f64>().unwrap_or(f64::NAN);
        let quote_volume = qi.and_then(|i| get(i).parse::<f64>().ok()).unwrap_or(0.0);
        let funding_rate = fi.and_then(|i| get(i).parse::<f64>().ok()).unwrap_or(0.0);
        let raw_target = ri.map(|i| get(i).parse::<f64>().unwrap_or(f64::NAN)).unwrap_or(f64::NAN);
        let funding_event_rate = ei.and_then(|i| get(i).parse::<f64>().ok())
            .filter(|v| v.is_finite()).unwrap_or(0.0);
        let row = Row {
            timestamp: get(ti).to_string(),
            symbol: get(si).to_string(),
            prediction,
            target,
            raw_target,
            quote_volume,
            funding_rate,
            funding_event_rate,
        };
        groups.entry(row.timestamp.clone()).or_default().push(row);
    }
    Ok((factor_id, columns, groups))
}

fn decile(rank: usize, n: usize) -> usize {
    if n <= 1 { return 1; }
    // Equivalent to the reference evaluator's rounded linear map from rank
    // to 1..10, with the endpoints always reserved for Q1 and Q10.
    1 + ((rank * 9 + (n - 1) / 2) / (n - 1)).min(9)
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() { None } else { Some(values.iter().sum::<f64>() / values.len() as f64) }
}

fn std(values: &[f64], avg: f64) -> f64 {
    if values.is_empty() { return 0.0; }
    (values.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
}

fn parse_hour_minute(timestamp: &str) -> Option<(u32, u32)> {
    let split = timestamp.find('T').or_else(|| timestamp.find(' '))?;
    let clock = timestamp.get(split + 1..)?;
    let mut parts = clock.split(':');
    Some((parts.next()?.parse().ok()?, parts.next()?.parse().ok()?))
}

fn json_number(value: f64) -> String {
    if value.is_finite() { format!("{value}") } else { "null".to_string() }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 1 || has_flag(&args, "--help") {
        usage();
        return Ok(());
    }
    let input = argument(&args, "--input").ok_or("--input is required")?;
    let output = argument(&args, "--output").ok_or("--output is required")?;
    let layers_output = argument(&args, "--layers-output");
    let timeseries_output = argument(&args, "--timeseries-output");
    let config = Config {
        long_percent: parse_f64(&args, "--long-percent", 10.0),
        short_percent: parse_f64(&args, "--short-percent", 10.0),
        signal_interval: parse_usize(&args, "--signal-interval-bars", 1).max(1),
        hold_bars: parse_usize(&args, "--signal-hold-bars", 1).max(1),
        gross_leverage: parse_f64(&args, "--gross-leverage", 1.0),
        initial_cash: parse_f64(&args, "--initial-cash", 10_000.0),
        include_costs: has_flag(&args, "--include-costs"),
        include_funding: has_flag(&args, "--include-funding"),
        // Matches crypto_quant.costs.CostModel.taker_fee_bps.
        taker_fee_bps: parse_f64(&args, "--taker-fee-bps", 5.0),
        slippage_base_bps: parse_f64(&args, "--slippage-base-bps", 1.0),
        slippage_impact_coef: parse_f64(&args, "--slippage-impact-coef", 60.0),
        max_slippage_bps: parse_f64(&args, "--max-slippage-bps", 80.0),
    };
    if config.gross_leverage <= 0.0 || config.long_percent < 0.0 || config.short_percent < 0.0
        || config.long_percent + config.short_percent > 100.0 {
        return Err("invalid leverage or long/short percentages".into());
    }

    let (factor_id, columns, groups) = read_predictions(&input)?;
    if !columns.has_raw_target {
        eprintln!("WARNING {input}: no raw_target column; portfolio return uses the market-demeaned target \
                   and is only approximate (re-export with scripts/export_factor_predictions.py)");
    }
    if config.include_funding && !columns.has_funding_events {
        eprintln!("WARNING {input}: no funding_event_rate column; funding is charged only at 00/08/16 UTC");
    }
    let mut previous: HashMap<String, f64> = HashMap::new();
    let mut current: HashMap<String, f64> = HashMap::new();
    let mut signal_bar: Option<usize> = None;
    let mut equity = config.initial_cash;
    let mut returns = Vec::new();
    let mut turnovers = Vec::new();
    let mut total_cost = 0.0;
    let mut signed_q1 = Vec::new();
    let mut signed_q10 = Vec::new();
    let mut abs_q1 = Vec::new();
    let mut abs_q10 = Vec::new();
    let mut abs_q10_directional = Vec::new();
    let mut layers = Vec::new();
    let mut timeseries = Vec::new();
    let mut peak_equity = equity;

    for (bar, (timestamp, rows)) in groups.iter().enumerate() {
        let mut valid: Vec<Row> = rows.iter()
            .filter(|r| r.prediction.is_finite() && r.target.is_finite()
                && (!columns.has_raw_target || r.raw_target.is_finite()))
            .cloned().collect();
        if valid.len() < 2 { continue; }
        valid.sort_by(|a, b| a.prediction.partial_cmp(&b.prediction).unwrap_or(std::cmp::Ordering::Equal));
        let n = valid.len();
        let mut signed_layers: [Vec<f64>; 10] = std::array::from_fn(|_| Vec::new());
        let mut abs_layers: [Vec<f64>; 10] = std::array::from_fn(|_| Vec::new());
        let mut abs_directional_layers: [Vec<f64>; 10] = std::array::from_fn(|_| Vec::new());
        for (rank, row) in valid.iter().enumerate() {
            let q = decile(rank, n);
            let abs_rank = {
                let mut order: Vec<(usize, f64)> = valid.iter().enumerate().map(|(i, r)| (i, r.prediction.abs())).collect();
                order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                order.iter().position(|(i, _)| *i == rank).unwrap_or(rank)
            };
            let aq = decile(abs_rank, n);
            signed_layers[q - 1].push(row.target);
            abs_layers[aq - 1].push(row.target);
            abs_directional_layers[aq - 1].push(row.prediction.signum() * row.target);
        }
        let signed_means: Vec<f64> = signed_layers.iter().map(|v| mean(v).unwrap_or(f64::NAN)).collect();
        let abs_means: Vec<f64> = abs_layers.iter().map(|v| mean(v).unwrap_or(f64::NAN)).collect();
        let abs_directional_means: Vec<f64> = abs_directional_layers.iter().map(|v| mean(v).unwrap_or(f64::NAN)).collect();
        if let (Some(a), Some(b)) = (mean(&signed_layers[0]), mean(&signed_layers[9])) { signed_q1.push(a); signed_q10.push(b); }
        if let (Some(a), Some(b)) = (mean(&abs_layers[0]), mean(&abs_layers[9])) {
            abs_q1.push(a); abs_q10.push(b);
            if let Some(v) = mean(&abs_directional_layers[9]) { abs_q10_directional.push(v); }
            let mut fields = vec![timestamp.clone()];
            fields.extend(signed_means.iter().map(|v| json_number(*v)));
            fields.extend(abs_means.iter().map(|v| json_number(*v)));
            fields.extend(abs_directional_means.iter().map(|v| json_number(*v)));
            layers.push(format!("{}\n", fields.join(",")));
        }

        if bar % config.signal_interval == 0 {
            current.clear();
            let nl = ((n as f64 * config.long_percent / 100.0).ceil() as usize).max(1).min(n);
            let ns = ((n as f64 * config.short_percent / 100.0).ceil() as usize).max(1).min(n);
            if nl + ns > n { return Err("long/short selections overlap".into()); }
            for row in valid.iter().rev().take(nl) { current.insert(row.symbol.clone(), config.gross_leverage / 2.0 / nl as f64); }
            for row in valid.iter().take(ns) { current.insert(row.symbol.clone(), -config.gross_leverage / 2.0 / ns as f64); }
            signal_bar = Some(bar);
        } else if let Some(start) = signal_bar {
            if bar.saturating_sub(start) >= config.hold_bars { current.clear(); signal_bar = None; }
        }

        let by_symbol: HashMap<&str, &Row> = valid.iter().map(|r| (r.symbol.as_str(), r)).collect();
        let mut symbols: HashSet<String> = previous.keys().cloned().collect();
        symbols.extend(current.keys().cloned());
        let mut turnover = 0.0;
        let mut costs = 0.0;
        if config.include_costs {
            for symbol in &symbols {
                let delta = current.get(symbol).copied().unwrap_or(0.0) - previous.get(symbol).copied().unwrap_or(0.0);
                turnover += delta.abs();
                if delta != 0.0 {
                    let notional = delta.abs() * equity;
                    let volume = by_symbol.get(symbol.as_str()).map(|r| r.quote_volume).unwrap_or(0.0);
                    let participation = if volume > 0.0 { notional / volume } else { 1.0 };
                    let slip = (config.slippage_base_bps + config.slippage_impact_coef * participation).min(config.max_slippage_bps);
                    costs += notional * (config.taker_fee_bps + slip) / 10_000.0;
                }
            }
        }
        // Portfolio return is the weighted sum of each asset's *simple* return.
        // Summing per-asset log returns and exponentiating the sum is not a
        // portfolio return: it adds ~0.5*Var(sum w*r) and rewards shorting
        // volatile coins (a zero-alpha 3x book gained >100%/yr that way).
        // With `raw_target` the return is exact; legacy CSVs fall back to the
        // market-demeaned `target`, which is close for a dollar-neutral book.
        let funding_now = config.include_funding && !columns.has_funding_events
            && parse_hour_minute(timestamp).map(|(h, m)| (h == 0 || h == 8 || h == 16) && m == 0).unwrap_or(false);
        let mut gross = 0.0;
        let mut funding = 0.0;
        for (symbol, weight) in &current {
            if let Some(row) = by_symbol.get(symbol.as_str()) {
                let log_return = if columns.has_raw_target { row.raw_target } else { row.target };
                gross += weight * log_return.exp_m1();
                if config.include_funding && columns.has_funding_events {
                    // Actual settlements inside this row's holding window.
                    funding += weight * equity * row.funding_event_rate;
                } else if funding_now {
                    funding += weight * equity * row.funding_rate;
                }
            }
        }
        let gross_log_return = gross.ln_1p();
        let cost_return = costs / equity.max(1e-12);
        let funding_return = funding / equity.max(1e-12);
        let net = gross - cost_return - funding_return;
        let equity_before = equity;
        equity *= 1.0 + net;
        let pnl = equity - equity_before;
        peak_equity = peak_equity.max(equity);
        let drawdown = equity / peak_equity - 1.0;
        returns.push(net);
        turnovers.push(turnover);
        total_cost += costs;
        timeseries.push(format!(
            "{timestamp},{gross_log_return},{gross},{net},{pnl},{equity},{drawdown},{turnover},{costs},{funding},{cost_return},{funding_return}\n"
        ));
        previous = current.clone();
    }

    let avg = mean(&returns).unwrap_or(0.0);
    let volatility = std(&returns, avg);
    let mut curve = config.initial_cash;
    let mut peak = curve;
    let mut max_drawdown: f64 = 0.0;
    for value in &returns { curve *= 1.0 + value; peak = peak.max(curve); max_drawdown = max_drawdown.min(curve / peak - 1.0); }
    let sharpe = if volatility > 0.0 { avg / volatility * (365.0 * 48.0_f64).sqrt() } else { f64::NAN };
    let signed_spread = mean(&signed_q10).unwrap_or(f64::NAN) - mean(&signed_q1).unwrap_or(f64::NAN);
    let abs_spread = mean(&abs_q10).unwrap_or(f64::NAN) - mean(&abs_q1).unwrap_or(f64::NAN);
    let json = format!(
        "{{\n  \"factor_id\": \"{}\",\n  \"target_basis\": \"log_excess_return\",\n  \"portfolio_return_basis\": \"{}\",\n  \"funding_basis\": \"{}\",\n  \"bars\": {},\n  \"final_equity\": {},\n  \"return_pct\": {},\n  \"sharpe\": {},\n  \"max_drawdown_pct\": {},\n  \"avg_turnover\": {},\n  \"total_cost\": {},\n  \"signed_q10_minus_q1\": {},\n  \"abs_q10_minus_q1\": {},\n  \"abs_q10_directional_return\": {}\n}}\n",
        factor_id,
        if columns.has_raw_target { "sum_w_simple_raw_return" } else { "sum_w_simple_excess_return_legacy_csv" },
        if columns.has_funding_events { "actual_settlements" } else { "legacy_00_08_16_utc" },
        returns.len(), json_number(equity),
        json_number((equity / config.initial_cash - 1.0) * 100.0),
        json_number(sharpe), json_number(max_drawdown * 100.0),
        json_number(mean(&turnovers).unwrap_or(0.0)), json_number(total_cost),
        json_number(signed_spread), json_number(abs_spread),
        json_number(mean(&abs_q10_directional).unwrap_or(f64::NAN))
    );
    fs::write(output, json)?;
    if let Some(path) = layers_output {
        let mut columns = vec!["timestamp".to_string()];
        columns.extend((1..=10).map(|q| format!("signed_q{q}")));
        columns.extend((1..=10).map(|q| format!("abs_q{q}")));
        columns.extend((1..=10).map(|q| format!("abs_q{q}_directional")));
        let mut body = format!("{}\n", columns.join(","));
        body.push_str(&layers.concat());
        fs::write(path, body)?;
    }
    if let Some(path) = timeseries_output {
        let mut body = String::from(
            "timestamp,gross_log_return,gross_return,net_return,pnl,equity,drawdown,turnover,cost,funding,cost_return,funding_return\n",
        );
        body.push_str(&timeseries.concat());
        fs::write(path, body)?;
    }
    println!("factor={} bars={} final_equity={:.6} signed_spread={:.8} abs_spread={:.8}", factor_id, returns.len(), equity, signed_spread, abs_spread);
    Ok(())
}
