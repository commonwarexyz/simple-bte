use ark_bls12_381::Bls12_381;
use ark_std::test_rng;
use simple_batched_threshold_encryption::bte::{
    crs::setup,
    fo::{
        batch_verify, combine, encrypt, helper_decrypt, helper_finalize, partial_decrypt,
        predecrypt_fft, verify_partial_decryption,
    },
};
use std::env;
use std::time::{Duration, Instant};

type E = Bls12_381;

#[derive(Clone, Copy, Debug)]
enum Mode {
    Direct,
    Pipelined,
}

#[derive(Clone, Debug)]
struct Config {
    batch_size: usize,
    num_parties: usize,
    threshold: usize,
    iterations: usize,
    mode: Mode,
    msg_len: usize,
}

#[derive(Default)]
struct Timing {
    setup: Duration,
    encrypt: Duration,
    partial_decrypt: Duration,
    verify_shares: Duration,
    combine: Duration,
    predecrypt: Duration,
    helper_finalize: Duration,
    helper_total: Duration,
    batch_verify: Duration,
}

fn main() {
    let config = parse_args();
    print_run_header(&config);

    let mut aggregate = Timing::default();

    for iter in 0..config.iterations {
        let mut rng = test_rng();

        // -- Setup --------------------------------------------------------
        let start = Instant::now();
        let (ek, dk, sks) =
            setup::<E>(config.batch_size, config.num_parties, config.threshold, &mut rng);
        let setup_time = start.elapsed();

        // -- Encrypt ------------------------------------------------------
        let messages: Vec<Vec<u8>> = (0..config.batch_size)
            .map(|i| {
                let mut msg = vec![0u8; config.msg_len];
                msg[..8.min(config.msg_len)].copy_from_slice(
                    &(i as u64).to_le_bytes()[..8.min(config.msg_len)],
                );
                msg
            })
            .collect();

        let start = Instant::now();
        let cts: Vec<_> = messages
            .iter()
            .map(|m| encrypt(&ek, m, &mut rng))
            .collect();
        let encrypt_time = start.elapsed();

        // -- Partial decrypt (per validator) ------------------------------
        let start = Instant::now();
        let pds: Vec<_> = sks[..config.threshold]
            .iter()
            .map(|sk| partial_decrypt(sk, &cts))
            .collect();
        let partial_decrypt_time = start.elapsed();

        // -- Verify shares ------------------------------------------------
        let start = Instant::now();
        for pd in &pds {
            assert!(
                verify_partial_decryption(&dk, pd, &cts),
                "share verification failed"
            );
        }
        let verify_shares_time = start.elapsed();

        // -- Combine ------------------------------------------------------
        let start = Instant::now();
        let pd = combine::<E>(&pds);
        let combine_time = start.elapsed();

        // -- Helper: expensive decrypt + hint generation ------------------
        let (predecrypt_time, helper_finalize_time, helper_total_time, helper_msgs, hints) =
            match config.mode {
                Mode::Direct => {
                    let start = Instant::now();
                    let (msgs, hints) = helper_decrypt(&dk, &pd, &cts);
                    let total = start.elapsed();
                    (Duration::ZERO, Duration::ZERO, total, msgs, hints)
                }
                Mode::Pipelined => {
                    let start = Instant::now();
                    let cross = predecrypt_fft(&dk, &cts);
                    let pre = start.elapsed();

                    let start = Instant::now();
                    let (msgs, hints) = helper_finalize(&dk, &pd, &cts, &cross);
                    let fin = start.elapsed();

                    (pre, fin, pre + fin, msgs, hints)
                }
            };

        for i in 0..config.batch_size {
            assert_eq!(helper_msgs[i], messages[i], "helper decrypt mismatch at {i}");
        }

        // -- Verifier: batch verify (no pairings!) ------------------------
        let start = Instant::now();
        let verified = batch_verify(&ek, &cts, &hints, &mut rng);
        let batch_verify_time = start.elapsed();

        let verified_msgs = verified.expect("batch verification failed");
        for i in 0..config.batch_size {
            assert_eq!(verified_msgs[i], messages[i], "verified message mismatch at {i}");
        }

        // -- Accumulate ---------------------------------------------------
        let t = Timing {
            setup: setup_time,
            encrypt: encrypt_time,
            partial_decrypt: partial_decrypt_time,
            verify_shares: verify_shares_time,
            combine: combine_time,
            predecrypt: predecrypt_time,
            helper_finalize: helper_finalize_time,
            helper_total: helper_total_time,
            batch_verify: batch_verify_time,
        };

        aggregate.setup += t.setup;
        aggregate.encrypt += t.encrypt;
        aggregate.partial_decrypt += t.partial_decrypt;
        aggregate.verify_shares += t.verify_shares;
        aggregate.combine += t.combine;
        aggregate.predecrypt += t.predecrypt;
        aggregate.helper_finalize += t.helper_finalize;
        aggregate.helper_total += t.helper_total;
        aggregate.batch_verify += t.batch_verify;

        print_iteration(iter + 1, config.mode, &t);
    }

    let avg = average_timing(&aggregate, config.iterations as u32);
    print_summary(config.mode, &avg);
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

fn parse_args() -> Config {
    let mut config = Config {
        batch_size: 2048,
        num_parties: 8,
        threshold: 5,
        iterations: 1,
        mode: Mode::Pipelined,
        msg_len: 256,
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--batch-size" | "-b" => {
                config.batch_size = parse_usize_arg("--batch-size", args.next());
            }
            "--num-parties" | "-n" => {
                config.num_parties = parse_usize_arg("--num-parties", args.next());
            }
            "--threshold" | "-t" => {
                config.threshold = parse_usize_arg("--threshold", args.next());
            }
            "--iters" | "-i" => {
                config.iterations = parse_usize_arg("--iters", args.next());
            }
            "--mode" | "-m" => {
                config.mode = parse_mode_arg(args.next());
            }
            "--msg-len" | "-l" => {
                config.msg_len = parse_usize_arg("--msg-len", args.next());
            }
            "--help" | "-h" => {
                print_help_and_exit();
            }
            other => {
                panic!("unknown argument: {other}");
            }
        }
    }

    assert!(
        config.threshold <= config.num_parties,
        "threshold must be <= num_parties"
    );
    assert!(config.iterations >= 1, "iters must be >= 1");
    config
}

fn parse_usize_arg(flag: &str, value: Option<String>) -> usize {
    value
        .unwrap_or_else(|| panic!("missing value for {flag}"))
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("invalid usize for {flag}"))
}

fn parse_mode_arg(value: Option<String>) -> Mode {
    match value
        .unwrap_or_else(|| panic!("missing value for --mode"))
        .as_str()
    {
        "direct" => Mode::Direct,
        "pipelined" => Mode::Pipelined,
        other => panic!("invalid mode: {other} (expected 'direct' or 'pipelined')"),
    }
}

fn print_help_and_exit() -> ! {
    println!("Usage: cargo run --release --example fo_e2e -- [options]");
    println!();
    println!("Options:");
    println!("  --batch-size, -b   Batch size (default: 2048)");
    println!("  --num-parties, -n  Number of parties (default: 8)");
    println!("  --threshold, -t    Threshold (default: 5)");
    println!("  --iters, -i        Iterations (default: 1)");
    println!("  --mode, -m         Helper mode: direct | pipelined (default: pipelined)");
    println!("  --msg-len, -l      Per-message byte length (default: 256)");
    std::process::exit(0);
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

fn average_timing(total: &Timing, n: u32) -> Timing {
    Timing {
        setup: total.setup / n,
        encrypt: total.encrypt / n,
        partial_decrypt: total.partial_decrypt / n,
        verify_shares: total.verify_shares / n,
        combine: total.combine / n,
        predecrypt: total.predecrypt / n,
        helper_finalize: total.helper_finalize / n,
        helper_total: total.helper_total / n,
        batch_verify: total.batch_verify / n,
    }
}

fn print_run_header(config: &Config) {
    println!("FO-BTE End-to-End (batch verification)");
    println!("======================================");
    println!("batch size : {}", config.batch_size);
    println!("num parties: {}", config.num_parties);
    println!("threshold  : {}", config.threshold);
    println!("msg length : {} bytes", config.msg_len);
    println!("iterations : {}", config.iterations);
    println!("mode       : {:?}", config.mode);
    println!();
}

fn print_iteration(iter: usize, mode: Mode, t: &Timing) {
    println!("Iteration {iter}");
    println!("-----------");
    print_timing_block(mode, t);
    println!();
}

fn print_summary(mode: Mode, avg: &Timing) {
    println!("Average");
    println!("-------");
    print_timing_block(mode, avg);
}

fn print_timing_block(mode: Mode, t: &Timing) {
    println!("setup             {}", fmt(t.setup));
    println!("encrypt           {}", fmt(t.encrypt));
    println!("partial_decrypt   {}", fmt(t.partial_decrypt));
    println!("verify_shares     {}", fmt(t.verify_shares));
    println!("combine           {}", fmt(t.combine));
    match mode {
        Mode::Pipelined => {
            println!("predecrypt (fft)  {}", fmt(t.predecrypt));
            println!("helper_finalize   {}", fmt(t.helper_finalize));
            println!("helper (total)    {}", fmt(t.helper_total));
        }
        Mode::Direct => {
            println!("helper_decrypt    {}", fmt(t.helper_total));
        }
    }
    println!("batch_verify      {}", fmt(t.batch_verify));
}

fn fmt(d: Duration) -> String {
    if d.as_secs() > 0 {
        format!("{:.3}s", d.as_secs_f64())
    } else if d.as_millis() > 0 {
        format!("{:.3}ms", d.as_secs_f64() * 1_000.0)
    } else if d.as_micros() > 0 {
        format!("{:.3}us", d.as_secs_f64() * 1_000_000.0)
    } else {
        format!("{}ns", d.as_nanos())
    }
}
