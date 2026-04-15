use ark_bls12_381::Bls12_381;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::PrimeGroup;
use ark_std::{test_rng, UniformRand};
use simple_batched_threshold_encryption::bte::{
    crs::setup,
    decryption::{combine, decrypt, decrypt_fft, partial_decrypt, verify, verify_ciphertext_batch},
    encryption::encrypt,
};
use std::env;
use std::time::{Duration, Instant};

type E = Bls12_381;
type Fr = <E as Pairing>::ScalarField;

#[derive(Clone, Copy, Debug)]
enum Mode {
    Naive,
    Fft,
}

#[derive(Clone, Debug)]
struct Config {
    batch_size: usize,
    num_parties: usize,
    threshold: usize,
    iterations: usize,
    mode: Mode,
}

#[derive(Default)]
struct Timing {
    setup: Duration,
    encrypt: Duration,
    verify_cts: Duration,
    partial_decrypt: Duration,
    verify_shares: Duration,
    combine: Duration,
    decrypt: Duration,
}

fn main() {
    let config = parse_args();
    print_run_header(&config);

    let mut aggregate = Timing::default();

    for iter in 0..config.iterations {
        let mut rng = test_rng();

        let start = Instant::now();
        let (ek, dk, sks) =
            setup::<E>(config.batch_size, config.num_parties, config.threshold, &mut rng);
        let setup_time = start.elapsed();

        let messages: Vec<PairingOutput<E>> = (0..config.batch_size)
            .map(|_| PairingOutput::<E>::generator() * Fr::rand(&mut rng))
            .collect();

        let start = Instant::now();
        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();
        let encrypt_time = start.elapsed();

        let start = Instant::now();
        let cts_ok = verify_ciphertext_batch(&cts, &mut rng);
        let verify_cts_time = start.elapsed();
        assert!(cts_ok, "ciphertext proof verification failed");

        let start = Instant::now();
        let pds: Vec<_> = sks[..config.threshold]
            .iter()
            .map(|sk| partial_decrypt(sk, &cts, &mut rng).expect("valid ciphertext proofs"))
            .collect();
        let partial_decrypt_time = start.elapsed();

        let start = Instant::now();
        for pd in &pds {
            assert!(verify(&dk, pd, &cts), "share verification failed");
        }
        let verify_shares_time = start.elapsed();

        let start = Instant::now();
        let pd = combine::<E>(&pds);
        let combine_time = start.elapsed();

        let start = Instant::now();
        let recovered = match config.mode {
            Mode::Naive => decrypt(&dk, &pd, &cts, &mut rng),
            Mode::Fft => decrypt_fft(&dk, &pd, &cts, &mut rng),
        };
        let decrypt_time = start.elapsed();

        assert_eq!(recovered.len(), messages.len());
        for i in 0..messages.len() {
            assert_eq!(recovered[i], messages[i], "decryption mismatch at {i}");
        }

        aggregate.setup += setup_time;
        aggregate.encrypt += encrypt_time;
        aggregate.verify_cts += verify_cts_time;
        aggregate.partial_decrypt += partial_decrypt_time;
        aggregate.verify_shares += verify_shares_time;
        aggregate.combine += combine_time;
        aggregate.decrypt += decrypt_time;

        print_iteration(
            iter + 1,
            &Timing {
                setup: setup_time,
                encrypt: encrypt_time,
                verify_cts: verify_cts_time,
                partial_decrypt: partial_decrypt_time,
                verify_shares: verify_shares_time,
                combine: combine_time,
                decrypt: decrypt_time,
            },
        );
    }

    let avg = average_timing(&aggregate, config.iterations as u32);
    print_summary(&avg);
}

fn parse_args() -> Config {
    let mut config = Config {
        batch_size: 2048,
        num_parties: 8,
        threshold: 5,
        iterations: 1,
        mode: Mode::Fft,
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
        "naive" => Mode::Naive,
        "fft" => Mode::Fft,
        other => panic!("invalid mode: {other} (expected 'naive' or 'fft')"),
    }
}

fn print_help_and_exit() -> ! {
    println!("Usage: cargo run --release --example decrypt_e2e -- [options]");
    println!("  --batch-size, -b   Batch size (default: 2048)");
    println!("  --num-parties, -n  Number of parties (default: 8)");
    println!("  --threshold, -t    Threshold (default: 5)");
    println!("  --iters, -i        Iterations (default: 1)");
    println!("  --mode, -m         Decrypt mode: naive | fft (default: fft)");
    std::process::exit(0);
}

fn average_timing(total: &Timing, n: u32) -> Timing {
    Timing {
        setup: total.setup / n,
        encrypt: total.encrypt / n,
        verify_cts: total.verify_cts / n,
        partial_decrypt: total.partial_decrypt / n,
        verify_shares: total.verify_shares / n,
        combine: total.combine / n,
        decrypt: total.decrypt / n,
    }
}

fn print_run_header(config: &Config) {
    println!("End-to-End Decryption");
    println!("=====================");
    println!("batch size : {}", config.batch_size);
    println!("num parties: {}", config.num_parties);
    println!("threshold  : {}", config.threshold);
    println!("iterations : {}", config.iterations);
    println!("mode       : {:?}", config.mode);
    println!();
}

fn print_iteration(iter: usize, timing: &Timing) {
    println!("Iteration {}", iter);
    println!("-----------");
    print_timing_block(timing);
    println!();
}

fn print_summary(avg: &Timing) {
    println!("Average");
    println!("-------");
    print_timing_block(avg);
}

fn print_timing_block(timing: &Timing) {
    println!("setup           {}", fmt_duration(timing.setup));
    println!("encrypt         {}", fmt_duration(timing.encrypt));
    println!("verify_cts      {}", fmt_duration(timing.verify_cts));
    println!("partial_decrypt {}", fmt_duration(timing.partial_decrypt));
    println!("verify_shares   {}", fmt_duration(timing.verify_shares));
    println!("combine         {}", fmt_duration(timing.combine));
    println!("decrypt         {}", fmt_duration(timing.decrypt));
}

fn fmt_duration(duration: Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{:.3}s", duration.as_secs_f64())
    } else if duration.as_millis() > 0 {
        format!("{:.3}ms", duration.as_secs_f64() * 1_000.0)
    } else if duration.as_micros() > 0 {
        format!("{:.3}us", duration.as_secs_f64() * 1_000_000.0)
    } else {
        format!("{}ns", duration.as_nanos())
    }
}
