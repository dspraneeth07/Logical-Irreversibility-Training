# Attempt 0, 2026-11-17: sensor validation refused

The first confirmatory session stopped during instrument validation, before any measurement window. The CPU energy counter raised its reading by 19.7 W under the reference load, but idle power varied so much between idle windows that the response was below the required three idle standard deviations, so the load-response test refused the sensor, as the preregistration requires.

Cause: another training job (a separate project's model training in a different Python environment, with four worker processes) was running on the same laptop. Idle package power measured right afterwards was 41.6 W with a standard deviation of 3.1 W, against 16 W in the GAI-001 sessions, with 2 to 3 cores busy in the background.

No energy data were collected, so no data are excluded. The confirmatory session is repeated from the start once the machine is quiet.

Part of [GAI-002](../../../README.md). Problem card: [GAI-002](../../../PROBLEM.md).
