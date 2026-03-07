use std::time::Instant;

use cubecl::frontend::CubePrimitive;
use cubecl::{Runtime, TestRuntime, future};
use cubefx_engine::{SignalSpec, phase_shift_effect};
use cubek_test_utils::{DataKind, Distribution, StrideSpec, TestInput};

fn main() {
    // Assumptions for the benchmarks and tests:
    // - channels will be 1 or 2
    // - sample_rate won't change
    // - window_length will always be a large power of 2
    // - hop_length always equals window_length / 2
    // Signal duration will however probably be larger
    let signal_spec = SignalSpec {
        signal_duration: 10.,
        channels: 2,
        sample_rate: 44100,
        window_length: 2048,
        hop_length: 1024,
    };

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked();

    let signal_handle = TestInput::new(
        client.clone(),
        signal_spec.signal_shape().to_vec(),
        dtype,
        StrideSpec::RowMajor,
        DataKind::Random {
            seed: 42,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_without_host_data();

    future::block_on(client.sync()).unwrap();

    let repeat = 10;
    let mut signal_processed = signal_handle.clone();

    // Warm-up run
    signal_processed = phase_shift_effect(signal_processed.clone(), 100., dtype);
    future::block_on(client.sync()).unwrap();
    println!("Warm-up run complete. Starting benchmark...");

    let mut total_duration = std::time::Duration::ZERO;

    for i in 0..repeat {
        let start = Instant::now();
        signal_processed = phase_shift_effect(signal_processed.clone(), 100., dtype);
        future::block_on(client.sync()).unwrap();
        let duration = start.elapsed();

        total_duration += duration;
        println!("Run {:2}: {:?}", i + 1, duration);
    }

    let avg = total_duration / repeat;
    println!(
        "Bench mode: ran {} repetitions. Avg per run: {:?}",
        repeat, avg,
    );
}
