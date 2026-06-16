# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Rayleigh Research

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import argparse
import bittensor as bt
from loguru import logger

from taos.common.config import add_validator_args

def add_im_validator_args(cls, parser):
    """Add validator specific arguments to the parser."""
    add_validator_args(cls, parser)
    
    parser.add_argument(
        "--repo.remote",
        type=str,
        help="Repository remote name.",
        default="origin",
    )
    
    parser.add_argument(
        '--benchmark.enabled',
        type=bool,
        default=False,
        help='Enable benchmark agents'
    )

    parser.add_argument(
        '--benchmark.agents',
        type=str,
        default='../config/benchmark_agents.json',
        help='JSON file path with benchmark agent configurations'
    )
    
    parser.add_argument(
        "--simulation.seeding.fundamental.symbol.coinbase",
        type=str,
        help="Coinbase spot market symbol price to be used to seed simulation price.",
        default="BTC-USD",
    )
    
    parser.add_argument(
        "--simulation.seeding.fundamental.symbol.binance",
        type=str,
        help="Binance spot market symbol price to be used to seed simulation price.",
        default="btcusdt",
    )
    
    parser.add_argument(
        "--simulation.seeding.external.symbol.coinbase",
        type=str,
        help="Coinbase futures market symbol price to be used to seed external price used in simulation.",
        default="TAO-PERP-INTX",
    )
    
    parser.add_argument(
        "--simulation.seeding.external.symbol.binance",
        type=str,
        help="Binance futures market symbol price to be used to seed external price used in simulation.",
        default="taousdt",
    )
    
    parser.add_argument(
        "--simulation.seeding.external.sampling_seconds",
        type=int,
        help="real time period in seconds over which external trade prices are written to file.",
        default=60,
    )

    parser.add_argument(
        "--simulation.xml_config",
        type=str,
        help="Path to XML file containing simulation configuration.",
        default="../../../simulate/trading/run/config/simulation_0.xml",
    )

    parser.add_argument(
        "--port",
        type=int,
        help="Port number on which to serve validator listener.",
        default=8000,
    )
    
    parser.add_argument(
        "--compression.engine",
        choices=['zlib', 'lz4', 'zstd'],
        help="Compression engine to apply, either `zlib` or `lz4` or `zstd`.",
        default="lz4",
    )
    
    parser.add_argument(
        "--compression.level",
        type=int,
        help="Compression level.",
        default=1,
    )

    parser.add_argument(
        "--compression.parallel_workers",
        type=int,
        help="Number of parallel workers to use in synapse compression. (0 => no parallelization, -1 => auto [half available cores])",
        default=-1,
    )
    
    parser.add_argument(
        "--scoring.interval",
        type=int,
        help="The simulation time interval at which reward calculation is executed.",
        default=5_000_000_000,
    )
    
    parser.add_argument(
        "--scoring.max_instructions_per_book",
        type=int,
        help="Maximum number of instructions that can be submitted by miners for each book in a single response.",
        default=5,
    )
    
    parser.add_argument(
        "--scoring.max_inactive_books",
        type=float,
        help="Maximum ratio of books that can be neglected without affecting score.  This number of books will be excluded from the scoring calculation (selected as lowest performing).",
        default=0.375,
    )

    parser.add_argument(
        "--scoring.kappa.weight",
        type=float,
        help="Weight applied to Kappa evaluation in final score calculation",
        default=0.79,
    )

    parser.add_argument(
        "--scoring.kappa.parallel_workers",
        type=int,
        help="Number of parallel workers to use in Kappa-3 calculation. (0 => no parallelization, -1 => auto [half available cores])",
        default=-1,
    )

    parser.add_argument(
        "--scoring.kappa.min_lookback",
        type=int,
        help="Minimum period of observations in simulation nanoseconds required for Kappa calculation.",
        default=5400_000_000_000,
    )

    parser.add_argument(
        "--scoring.kappa.lookback",
        type=int,
        help="Window in simulation nanoseconds of realized P&L observations to use for Kappa-3 ratio calculation.",
        default=10800_000_000_000,
    )

    parser.add_argument(
        "--scoring.kappa.tau",
        type=float,
        help="Threshold return parameter for Kappa-3 calculation (minimum acceptable return per period).",
        default=0.0,
    )
    
    parser.add_argument(
        "--scoring.kappa.min_realized_observations",
        type=int,
        help="The minimum number of realized P&L observations (round-trips) required in the assessment window for Kappa-3 score to be assigned.",
        default=3,
    )

    parser.add_argument(
        "--scoring.kappa.normalization_min",
        type=float,
        help="Kappa-3 values are normalized to fall within a range so as to produce non-negative value and facilitate scoring calculations. This is the minimum value in the normalization range.",
        default=-2.5,
    )

    parser.add_argument(
        "--scoring.kappa.normalization_max",
        type=float,
        help="Kappa-3 values are normalized to fall within a range so as to produce non-negative value and facilitate scoring calculations. This is the maximum value in the normalization range.",
        default=2.5,
    )
    
    parser.add_argument(
        "--scoring.kappa.pnl.impact",
        type=float,
        help="Multiplied onto normalized Kappa-3 values to modify the impact of realized PnL in scoring calculations.",
        default=0.0,
    )

    parser.add_argument(
        "--scoring.pnl.weight",
        type=float,
        help="Weight applied to Realized PnL evaluation in final score calculation",
        default=0.21,
    )

    parser.add_argument(
        "--scoring.pnl.normalization.method",
        type=str,
        help="Method for normalizing P&L: 'daily_return'",
        default="daily_return",
    )

    parser.add_argument(
        "--scoring.pnl.normalization.min_daily_return",
        type=float,
        help="Floor for daily return ratio.",
        default=-1.0,
    )

    parser.add_argument(
        "--scoring.pnl.normalization.max_daily_return",
        type=float,
        help="Cap for daily return ratio.",
        default=1.0,
    )

    parser.add_argument(
        "--scoring.gentrx.simulation_share",
        type=float,
        help="Share of miner rewards reserved for GenTRX gradient submitters. "
             "The default 0.05 means rewards split 95%% to trading "
             "(kappa+pnl) and up to 5%% to training, scaled by participation "
             "(N_active / N_registered_miners). The unused training portion "
             "returns to trading. When GenTRX is not running, no gradients "
             "are submitted and 100%% of rewards go to trading regardless "
             "of this setting.",
        default=0.05,
    )

    parser.add_argument(
        "--scoring.gentrx.ema_alpha",
        type=float,
        help="Per-UID EMA alpha applied inside score_uid to smooth the "
             "rank-normalized gentrx score across rounds before the slow "
             "validator-level moving average. Smaller = more smoothing.",
        default=0.1,
    )

    # ---- GenTRX distributed training ----
    # GenTRX is now HTTP-only — the gradient server runs as a separate process
    # (typically a sibling on the same host for single-machine setups, talking
    # over loopback). All aggregator / scoring / book-distribution tunables
    # live on the standalone gradient server's CLI; the validator side just
    # configures how to reach it.
    parser.add_argument(
        "--gentrx.enabled",
        action="store_true",
        help="Enable GenTRX: push sim state to the gradient server, deliver "
             "assignments to miners via dendrite, expose scores to weight calc.",
        default=False,
    )
    parser.add_argument(
        "--gentrx.gradient_server_url",
        type=str,
        help="Gradient server base URL (e.g. http://127.0.0.1:8100/gentrx for "
             "single-machine setups). REQUIRED when --gentrx.enabled is set.",
        default="",
    )
    parser.add_argument(
        "--gentrx.api_key",
        type=str,
        help="Shared secret for validator↔gradient server auth (also "
             "GENTRX_API_KEY env var). Required when the gradient server "
             "binds to a non-loopback interface.",
        default="",
    )
    parser.add_argument(
        "--gentrx.interval",
        type=int,
        help="Poll interval in seconds for score polls and round cadence in "
             "timer mode (blocks_per_round=0). In block-synced mode, round "
             "cadence is driven by the chain.",
        default=30,
    )
    parser.add_argument(
        "--gentrx.blocks_per_round",
        type=int,
        help="Block-synced round cadence: round = block // blocks_per_round. "
             "The validator derives the round from the chain and pushes "
             "POST /gentrx/round to the gradient server — the server itself "
             "has no block-sync config. Default 25 ≈ 5min at mainnet 12s/block, "
             "matches the 5min training window. Pass 0 for timer mode (proxy only).",
        default=25,
    )
    parser.add_argument(
        "--scoring.activity.trade_volume_sampling_interval",
        type=int,
        help="The simulation time interval at which miner agent trading volume history is sampled.",
        default=600_000_000_000,
    )
    
    parser.add_argument(
        "--scoring.activity.trade_volume_assessment_period",
        type=int,
        help="The period in simulation timesteps over which agent trading volumes are aggregated when evaluating activity.",
        default=86400_000_000_000,
    )
    
    parser.add_argument(
        "--scoring.activity.impact",
        type=float,
        help="Volume boost above neutral activity (1.0). At 0.0, any round-trip in the "
             "last trade_volume_sampling_interval yields activity factor 1.0; higher volume "
             "does not boost further.",
        default=0.0,
    )
    
    parser.add_argument(
        "--scoring.activity.decay_grace_period",
        type=int,
        help="Simulation nanoseconds after the last counted round-trip before activity-factor "
             "decay accelerates. Matches trade_volume_sampling_interval (10min) so missing "
             "a round-trip in the current window triggers decay immediately.",
        default=600_000_000_000,
    )
    
    parser.add_argument(
        "--scoring.activity.decay_rate",
        type=float,
        help="Decay rate applied per scoring tick when no round-trip closed in the last "
             "trade_volume_sampling_interval (10min). Must be >0 so activity factor 1.0 "
             "requires a recent round-trip; 0 disables decay and lets inactive books coast.",
        default=1.0,
    )

    parser.add_argument(
        "--scoring.activity.capital_turnover_cap",
        type=float,
        help="The number of times within each `trade_volume_assessment_period` that miner agents are able to trade the equivalent in volume to their initial capital allocation value before they are restricted from further activity.",
        default=10.0,
    )

    parser.add_argument(
        "--scoring.inventory.min_balance_ratio_multiplier",
        type=float,
        help="The minimum value for the multiplier applied to Kappa-3 scores to penalize holding of small ratio of BASE currency.",
        default=0.5,
    )

    parser.add_argument(
        "--scoring.inventory.max_balance_ratio_multiplier",
        type=float,
        help="The maximum value for the multiplier applied to Kappa-3 scores to reward holding larger ratio of BASE currency.",
        default=1.2,
    )

    parser.add_argument(
        "--scoring.min_delay",
        type=int,
        help="Minimum simulation timestamp delay that may be applied to miner responses.",
        default=10_000_000,
    )

    parser.add_argument(
        "--scoring.max_delay",
        type=int,
        help="Maximum simulation timestamp delay to may be applied to miner responses.",
        default=1000_000_000,
    )

    parser.add_argument(
        "--scoring.min_instruction_delay",
        type=int,
        help="Minimum additive simulation timestamp delay to be applied to subsequent instructions sent in the same response.",
        default=5_000_000,
    )

    parser.add_argument(
        "--scoring.max_instruction_delay",
        type=int,
        help="Maximum additive simulation timestamp delay to be applied to subsequent instructions sent in the same response.",
        default=25_000_000,
    )

    parser.add_argument(
        "--rewarding.seed",
        type=int,
        help="Seed to use in generating distribution for rewards.",
        default=898746039182,
    )

    parser.add_argument(
        "--rewarding.pareto.scale",
        type=float,
        help="Scale parameter for Pareto distribution used in allocating rewards.",
        default=1.0,
    )

    parser.add_argument(
        "--rewarding.pareto.shape",
        type=float,
        help="Shape parameter for Pareto distribution used in allocating rewards.",
        default=1.42,
    )
    
    parser.add_argument(
        "--reporting.disabled",
        action="store_true",
        help="If set, the validator will not publish metrics.",
        default=False,
    )

    # ── Exchange engine arguments ──────────────────────────────────────────────
    parser.add_argument(
        "--exchange.netuids",
        type=str,
        help="Comma-separated subnet UIDs to include as exchange books. "
             "Empty = auto-discover from chain state on first tick.",
        default="",
    )

    parser.add_argument(
        "--exchange.wallet.mode",
        type=str,
        choices=["single", "per_agent"],
        help="Wallet mode for on-chain execution. "
             "'single': one default wallet for all agents; "
             "'per_agent': each agent UID uses a dedicated wallet.",
        default="single",
    )

    parser.add_argument(
        "--exchange.wallet.path",
        type=str,
        help="Filesystem path to the bittensor wallets directory.",
        default="~/.bittensor/wallets",
    )

    parser.add_argument(
        "--exchange.timeout",
        type=float,
        help="IPC response timeout in seconds for LOB exchange communication.",
        default=60.0,
    )

    parser.add_argument(
        "--exchange.max_retries",
        type=int,
        help="Maximum number of send+receive attempts before giving up on a batch.",
        default=3,
    )

    parser.add_argument(
        "--exchange.ipc.request_queue",
        type=str,
        help="POSIX message queue name for LOB batch requests.",
        default="/mvtrx_req_queue",
    )

    parser.add_argument(
        "--exchange.ipc.response_queue",
        type=str,
        help="POSIX message queue name for LOB batch responses.",
        default="/mvtrx_res_queue",
    )

    parser.add_argument(
        "--exchange.ipc.request_shm",
        type=str,
        help="POSIX shared memory name for LOB batch request payloads.",
        default="/mvtrx_req_shm",
    )

    parser.add_argument(
        "--exchange.ipc.response_shm",
        type=str,
        help="POSIX shared memory name for LOB batch response payloads.",
        default="/mvtrx_res_shm",
    )

    parser.add_argument(
        "--exchange.ipc.request_semaphore",
        type=str,
        help="POSIX semaphore name used to signal the LOB that a request is ready.",
        default="/mvtrx_req_sem",
    )

    parser.add_argument(
        "--exchange.ipc.response_semaphore",
        type=str,
        help="POSIX semaphore name used by the LOB to signal that a response is ready.",
        default="/mvtrx_res_sem",
    )
