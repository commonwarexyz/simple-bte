use ark_bls12_381::Bls12_381;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::Zero;
use ark_poly::EvaluationDomain;
use ark_std::{rand::Rng, test_rng, UniformRand};
use simple_batched_threshold_encryption::bte::{
    crs::setup,
    fo::{
        batch_verify, batch_verify_bandwidth_optimized, combine, encrypt, helper_decrypt,
        helper_decrypt_bandwidth_optimized, helper_finalize, helper_finalize_bandwidth_optimized,
        partial_decrypt, predecrypt_fft, verify_partial_decryption, DecryptionHints, FoCiphertext,
    },
    DecryptionKey,
};
use std::env;
use std::time::{Duration, Instant};

type E = Bls12_381;
type Fr = <E as Pairing>::ScalarField;

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
    helper_finalize_verification: Duration,
    helper_finalize_bandwidth: Duration,
    helper_total_verification: Duration,
    helper_total_bandwidth: Duration,
    naive_batch_verify: Duration,
    batch_verify_verification: Duration,
    batch_verify_bandwidth: Duration,
}

fn main() {
    let config = parse_args();
    print_run_header(&config);

    let mut aggregate = Timing::default();

    for iter in 0..config.iterations {
        let mut rng = test_rng();

        // -- Setup --------------------------------------------------------
        let start = Instant::now();
        let (ek, dk, sks) = setup::<E>(
            config.batch_size,
            config.num_parties,
            config.threshold,
            &mut rng,
        );
        let setup_time = start.elapsed();

        // -- Encrypt ------------------------------------------------------
        let messages: Vec<Vec<u8>> = (0..config.batch_size)
            .map(|i| {
                let mut msg = vec![0u8; config.msg_len];
                msg[..8.min(config.msg_len)]
                    .copy_from_slice(&(i as u64).to_le_bytes()[..8.min(config.msg_len)]);
                msg
            })
            .collect();

        let start = Instant::now();
        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();
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
        let naive_h_hat = naive_s_fft_kernel(&dk);

        // -- Helper: expensive decrypt + hint generation ------------------
        let (
            predecrypt_time,
            helper_finalize_verification_time,
            helper_finalize_bandwidth_time,
            helper_total_verification_time,
            helper_total_bandwidth_time,
            helper_msgs,
            helper_bandwidth_msgs,
            hints,
            bandwidth_hints,
        ) = match config.mode {
            Mode::Direct => {
                let start = Instant::now();
                let (helper_msgs, hints) = helper_decrypt(&dk, &pd, &cts);
                let verification_total = start.elapsed();

                let start = Instant::now();
                let (helper_bandwidth_msgs, bandwidth_hints) =
                    helper_decrypt_bandwidth_optimized(&dk, &pd, &cts);
                let bandwidth_total = start.elapsed();

                (
                    Duration::ZERO,
                    Duration::ZERO,
                    Duration::ZERO,
                    verification_total,
                    bandwidth_total,
                    helper_msgs,
                    helper_bandwidth_msgs,
                    hints,
                    bandwidth_hints,
                )
            }
            Mode::Pipelined => {
                let start = Instant::now();
                let cross = predecrypt_fft(&dk, &cts);
                let pre = start.elapsed();

                let start = Instant::now();
                let (helper_msgs, hints) = helper_finalize(&dk, &pd, &cts, &cross);
                let verification_finalize = start.elapsed();

                let start = Instant::now();
                let (helper_bandwidth_msgs, bandwidth_hints) =
                    helper_finalize_bandwidth_optimized(&dk, &pd, &cts, &cross);
                let bandwidth_finalize = start.elapsed();

                (
                    pre,
                    verification_finalize,
                    bandwidth_finalize,
                    pre + verification_finalize,
                    pre + bandwidth_finalize,
                    helper_msgs,
                    helper_bandwidth_msgs,
                    hints,
                    bandwidth_hints,
                )
            }
        };

        for i in 0..config.batch_size {
            assert_eq!(
                helper_msgs[i], messages[i],
                "helper decrypt mismatch at {i}"
            );
            assert_eq!(
                helper_bandwidth_msgs[i], messages[i],
                "bandwidth helper decrypt mismatch at {i}"
            );
        }

        // -- Verifier: naive randomized pairing check ---------------------
        let start = Instant::now();
        assert!(
            naive_batch_verify_decryption(&dk, &pd, &cts, &hints, &naive_h_hat, &mut rng),
            "naive batch verification failed"
        );
        let naive_batch_verify_time = start.elapsed();

        // -- Verifier: batch verify (no pairings!) ------------------------
        let start = Instant::now();
        let verified = batch_verify(&ek, &cts, &hints, &mut rng);
        let batch_verify_verification_time = start.elapsed();

        let verified_msgs = verified.expect("batch verification failed");
        for i in 0..config.batch_size {
            assert_eq!(
                verified_msgs[i], messages[i],
                "verified message mismatch at {i}"
            );
        }

        let start = Instant::now();
        let verified_bandwidth =
            batch_verify_bandwidth_optimized(&ek, &cts, &bandwidth_hints, &mut rng);
        let batch_verify_bandwidth_time = start.elapsed();

        let verified_bandwidth_msgs =
            verified_bandwidth.expect("bandwidth batch verification failed");
        for i in 0..config.batch_size {
            assert_eq!(
                verified_bandwidth_msgs[i], messages[i],
                "bandwidth verified message mismatch at {i}"
            );
        }

        // -- Accumulate ---------------------------------------------------
        let t = Timing {
            setup: setup_time,
            encrypt: encrypt_time,
            partial_decrypt: partial_decrypt_time,
            verify_shares: verify_shares_time,
            combine: combine_time,
            predecrypt: predecrypt_time,
            helper_finalize_verification: helper_finalize_verification_time,
            helper_finalize_bandwidth: helper_finalize_bandwidth_time,
            helper_total_verification: helper_total_verification_time,
            helper_total_bandwidth: helper_total_bandwidth_time,
            naive_batch_verify: naive_batch_verify_time,
            batch_verify_verification: batch_verify_verification_time,
            batch_verify_bandwidth: batch_verify_bandwidth_time,
        };

        aggregate.setup += t.setup;
        aggregate.encrypt += t.encrypt;
        aggregate.partial_decrypt += t.partial_decrypt;
        aggregate.verify_shares += t.verify_shares;
        aggregate.combine += t.combine;
        aggregate.predecrypt += t.predecrypt;
        aggregate.helper_finalize_verification += t.helper_finalize_verification;
        aggregate.helper_finalize_bandwidth += t.helper_finalize_bandwidth;
        aggregate.helper_total_verification += t.helper_total_verification;
        aggregate.helper_total_bandwidth += t.helper_total_bandwidth;
        aggregate.naive_batch_verify += t.naive_batch_verify;
        aggregate.batch_verify_verification += t.batch_verify_verification;
        aggregate.batch_verify_bandwidth += t.batch_verify_bandwidth;

        print_iteration(iter + 1, config.mode, &t);
    }

    let avg = average_timing(&aggregate, config.iterations as u32);
    print_summary(config.mode, &avg);
}

// ---------------------------------------------------------------------------
// Naive pairing-based batch verification
// ---------------------------------------------------------------------------

fn naive_s_fft_kernel(dk: &DecryptionKey<E>) -> Vec<<E as Pairing>::G2> {
    let b = dk.batch_size;
    let n = dk.fft_size;

    let mut h_vec = vec![<E as Pairing>::G2::zero(); n];
    h_vec[0] = dk.powers_of_h_affine[b + 1].into_group();
    for d in 1..b {
        h_vec[d] = dk.powers_of_h_affine[b + 1 + d].into_group();
        h_vec[n - d] = dk.powers_of_h_affine[b + 1 - d].into_group();
    }

    dk.fft_domain.fft_in_place(&mut h_vec);
    h_vec
}

fn compute_naive_s_fft(
    dk: &DecryptionKey<E>,
    h_hat: &[<E as Pairing>::G2],
    r: &[Fr],
) -> Vec<<E as Pairing>::G2Affine> {
    let b = dk.batch_size;
    let n = dk.fft_size;
    assert_eq!(r.len(), b);

    let mut r_hat = r.to_vec();
    r_hat.resize(n, Fr::zero());
    dk.fft_domain.fft_in_place(&mut r_hat);

    let mut s_hat = h_hat.to_vec();
    for (s, r_i) in s_hat.iter_mut().zip(r_hat) {
        *s *= r_i;
    }

    dk.fft_domain.ifft_in_place(&mut s_hat);

    <E as Pairing>::G2::normalize_batch(&s_hat[..b])
}

fn naive_batch_verify_decryption(
    dk: &DecryptionKey<E>,
    pd: &<E as Pairing>::G1,
    cts: &[FoCiphertext<E>],
    hints: &DecryptionHints<E>,
    h_hat: &[<E as Pairing>::G2],
    rng: &mut impl Rng,
) -> bool {
    let b = cts.len();
    assert_eq!(hints.pairing_values.len(), b);

    let r: Vec<Fr> = (0..b).map(|_| Fr::rand(rng)).collect();

    let mut lhs = PairingOutput::<E>::zero();
    for i in 0..b {
        lhs += hints.pairing_values[i] * r[i];
    }

    let pd_bases: Vec<_> = (0..b).map(|i| dk.powers_of_h_affine[b - i]).collect();
    let pd_term_g2 = <E as Pairing>::G2::msm(&pd_bases, &r).unwrap();
    let pd_term = E::pairing(*pd, pd_term_g2.into_affine());

    let s_values = compute_naive_s_fft(dk, h_hat, &r);
    let cross = E::multi_pairing(cts.iter().map(|ct| ct.ct0), s_values);

    lhs == pd_term - cross
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
            "--hint-mode" => {
                parse_hint_mode_arg(args.next());
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

fn parse_hint_mode_arg(value: Option<String>) {
    match value
        .unwrap_or_else(|| panic!("missing value for --hint-mode"))
        .as_str()
    {
        "verification" | "bandwidth" => {}
        other => panic!("invalid hint mode: {other} (expected 'verification' or 'bandwidth')"),
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
    println!("  --hint-mode        Deprecated; both verification strategies now run");
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
        helper_finalize_verification: total.helper_finalize_verification / n,
        helper_finalize_bandwidth: total.helper_finalize_bandwidth / n,
        helper_total_verification: total.helper_total_verification / n,
        helper_total_bandwidth: total.helper_total_bandwidth / n,
        naive_batch_verify: total.naive_batch_verify / n,
        batch_verify_verification: total.batch_verify_verification / n,
        batch_verify_bandwidth: total.batch_verify_bandwidth / n,
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
    println!("strategies : naive, bandwidth, verification");
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
            println!("predecrypt (fft)                {}", fmt(t.predecrypt));
            println!(
                "helper_finalize (verification)  {}",
                fmt(t.helper_finalize_verification)
            );
            println!(
                "helper_finalize (bandwidth)     {}",
                fmt(t.helper_finalize_bandwidth)
            );
            println!(
                "helper total (verification)     {}",
                fmt(t.helper_total_verification)
            );
            println!(
                "helper total (bandwidth)        {}",
                fmt(t.helper_total_bandwidth)
            );
        }
        Mode::Direct => {
            println!(
                "helper_decrypt (verification)   {}",
                fmt(t.helper_total_verification)
            );
            println!(
                "helper_decrypt (bandwidth)      {}",
                fmt(t.helper_total_bandwidth)
            );
        }
    }
    println!(
        "batch_verify (naive)           {}",
        fmt(t.naive_batch_verify)
    );
    println!(
        "batch_verify (verification)     {}",
        fmt(t.batch_verify_verification)
    );
    println!(
        "batch_verify (bandwidth)        {}",
        fmt(t.batch_verify_bandwidth)
    );
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
