#!/bin/bash
# ============================================================================
# G1 Real Robot Deployment with PICO 4 Teleoperation
# ============================================================================
# This script runs the C++ deployment on REAL G1 robot hardware,
# receiving control commands from PICO 4 VR headset via XRoboToolkit.
#
# Architecture:
#   INPUT:  PICO 4 → XRoboToolkit SDK → ZMQ (5556) → g1_deploy_onnx_ref
#   OUTPUT: g1_deploy_onnx_ref → ROS2 → DDS → G1 Robot (Unitree SDK2)
#
# Prerequisites:
#   1. XRoboToolkit PC Service installed and running
#   2. PICO 4 connected to same network
#   3. G1 robot connected via ethernet (192.168.123.x network)
#   4. Docker image built: g1-deploy-tensorrt10:latest
#
# Usage:
#   ./run_docker_real.sh [OPTIONS]
#
# Options:
#   --interface, -i    Network interface for robot (default: auto-detect 192.168.123.x)
#   --zmq-port, -z     ZMQ port for PICO commands (default: 5556)
#   --pico-host        PICO headset IP address (default: localhost via XRoboToolkit)
#   --no-prompt        Skip confirmation prompt
#   --help, -h         Show this help message
#
# Example:
#   ./run_docker_real.sh                    # Auto-detect interface
#   ./run_docker_real.sh -i enP8p1s0        # Specify interface
#   ./run_docker_real.sh --no-prompt        # No confirmation
#
# Related Scripts:
#   - XRoboToolkit PC Service must be running
#   - PICO teleop: ~/workspace/sonic/XRoboToolkit-Teleop-Sample-Python/scripts/simulation/teleop_unitree_g1_placo.py
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# Default Configuration
# ============================================================================

# Network interface (auto-detected by default)
NETWORK_INTERFACE=""

# ZMQ configuration
ZMQ_PORT="5556"
ZMQ_HOST="localhost"

# PICO configuration
PICO_HOST="localhost"

# Model paths
DECODER_MODEL="policy/release/model_decoder.onnx"
ENCODER_MODEL="policy/release/model_encoder.onnx"
OBS_CONFIG="policy/release/observation_config.yaml"
PLANNER_MODEL="planner/target_vel/V2/planner_sonic.onnx"
MOTION_DATA="reference/example/"

# Skip confirmation
NO_PROMPT=false

# ============================================================================
# Helper Functions
# ============================================================================

# Get network interface with 192.168.123.x IP (Unitree robot network)
detect_robot_interface() {
    local interface=""
    
    # Try ip command first (Linux)
    if command -v ip &> /dev/null; then
        interface=$(ip -4 addr show 2>/dev/null | grep -A2 "192.168.123" | grep -oP '^\d+:\K[^:]+' | head -1)
    fi
    
    # Fallback to ifconfig
    if [[ -z "$interface" ]] && command -v ifconfig &> /dev/null; then
        interface=$(ifconfig 2>/dev/null | grep -B1 "192.168.123" | grep -oP '^[^:]+' | head -1)
    fi
    
    # Common interface names for Unitree robots
    if [[ -z "$interface" ]]; then
        for iface in enP8p1s0 enp5s0 eth0 eth1 enp0s31f6; do
            if ip link show "$iface" &> /dev/null 2>&1; then
                interface="$iface"
                break
            fi
        done
    fi
    
    echo "$interface"
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run G1 robot deployment with PICO 4 teleoperation."
    echo ""
    echo "Options:"
    echo "  -i, --interface IFACE   Network interface for robot (default: auto-detect)"
    echo "  -z, --zmq-port PORT     ZMQ port for PICO commands (default: 5556)"
    echo "  --pico-host HOST        PICO headset IP (default: localhost)"
    echo "  --no-prompt             Skip confirmation prompt"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                      # Auto-detect interface"
    echo "  $0 -i enP8p1s0           # Specify interface"
    echo "  $0 --no-prompt           # No confirmation"
    echo ""
    echo "Prerequisites:"
    echo "  1. XRoboToolkit PC Service running"
    echo "  2. PICO 4 on same network"
    echo "  3. G1 robot connected (192.168.123.x)"
}

# ============================================================================
# Parse Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interface)
            NETWORK_INTERFACE="$2"
            shift 2
            ;;
        -z|--zmq-port)
            ZMQ_PORT="$2"
            shift 2
            ;;
        --pico-host)
            PICO_HOST="$2"
            shift 2
            ;;
        --no-prompt)
            NO_PROMPT=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# Detect Network Interface
# ============================================================================

if [[ -z "$NETWORK_INTERFACE" ]]; then
    echo -e "${BLUE}[Network Detection]${NC} Auto-detecting robot interface..."
    NETWORK_INTERFACE=$(detect_robot_interface)
    
    if [[ -z "$NETWORK_INTERFACE" ]]; then
        echo -e "${RED}❌ Could not detect robot network interface${NC}"
        echo "   Please specify manually: $0 -i <interface>"
        echo ""
        echo "   Common interfaces: enP8p1s0, enp5s0, eth0"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Detected interface: $NETWORK_INTERFACE${NC}"
fi

# ============================================================================
# Display Header
# ============================================================================

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          G1 REAL ROBOT DEPLOYMENT - PICO 4 TELEOPERATION            ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# Check Prerequisites
# ============================================================================

echo -e "${BLUE}[Prerequisites Check]${NC}"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker installed${NC}"

# Check Docker image
if ! docker images | grep -q "g1-deploy-tensorrt10"; then
    echo -e "${YELLOW}⚠️  Docker image 'g1-deploy-tensorrt10:latest' not found${NC}"
    echo "   Build it first: docker build -t g1-deploy-tensorrt10:latest -f Dockerfile.tensorrt10 ."
fi
echo -e "${GREEN}✅ Docker image available${NC}"

# Check model files
check_file() {
    if [[ ! -f "$1" ]]; then
        echo -e "${RED}❌ Missing: $1${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Found: $1${NC}"
        return 0
    fi
}

echo ""
echo "Checking model files..."
check_file "$DECODER_MODEL" || exit 1
check_file "$ENCODER_MODEL" || exit 1
check_file "$OBS_CONFIG" || exit 1
check_file "$PLANNER_MODEL" || exit 1

if [[ ! -d "$MOTION_DATA" ]]; then
    echo -e "${RED}❌ Missing directory: $MOTION_DATA${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Found: $MOTION_DATA${NC}"

# Check XRoboToolkit (optional - may run on different machine)
echo ""
echo -e "${YELLOW}ℹ️  XRoboToolkit PC Service should be running for PICO input${NC}"
echo "   Start with: XRoboToolkit PC Service (from GitHub releases)"

# ============================================================================
# Display Configuration
# ============================================================================

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                      DEPLOYMENT CONFIGURATION                         ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${YELLOW}⚠️  REAL ROBOT MODE - Safety checks ENABLED${NC}"
echo ""
echo -e "  Network Interface:  ${GREEN}$NETWORK_INTERFACE${NC}"
echo -e "  ZMQ Port:           ${GREEN}$ZMQ_PORT${NC}"
echo -e "  ZMQ Host:           ${GREEN}$ZMQ_HOST${NC}"
echo -e "  PICO Host:          ${GREEN}$PICO_HOST${NC}"
echo ""
echo -e "  Decoder Model:      ${GREEN}$DECODER_MODEL${NC}"
echo -e "  Encoder Model:      ${GREEN}$ENCODER_MODEL${NC}"
echo -e "  Obs Config:         ${GREEN}$OBS_CONFIG${NC}"
echo -e "  Planner Model:      ${GREEN}$PLANNER_MODEL${NC}"
echo -e "  Motion Data:        ${GREEN}$MOTION_DATA${NC}"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Architecture:${NC}"
echo -e "  ${GREEN}INPUT:${NC}  PICO 4 → XRoboToolkit SDK → ZMQ:${ZMQ_PORT} → g1_deploy_onnx_ref"
echo -e "  ${GREEN}OUTPUT:${NC} g1_deploy_onnx_ref → ROS2 → DDS → G1 Robot (Unitree SDK2)"
echo ""

# ============================================================================
# Confirmation
# ============================================================================

if [[ "$NO_PROMPT" != true ]]; then
    echo -e "${RED}⚠️  WARNING: This will control the REAL G1 robot!${NC}"
    echo -e "${YELLOW}   Ensure robot is in safe state and E-stop is accessible.${NC}"
    echo ""
    read -p "$(echo -e ${GREEN}Proceed with real robot deployment? [y/N]: ${NC})" confirm
    
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Deployment cancelled.${NC}"
        exit 0
    fi
fi

# ============================================================================
# Stop Existing Container
# ============================================================================

echo ""
echo -e "${BLUE}[Starting Deployment]${NC}"
echo ""

docker stop g1-real 2>/dev/null || true

# ============================================================================
# Run Docker Container
# ============================================================================

echo -e "${GREEN}🚀 Starting G1 Deploy container...${NC}"
echo ""

# Run with:
# - GPU access for TensorRT
# - Host network for DDS communication
# - Real-time priority for low latency
# - NO --disable-crc-check (safety enabled)

docker run -it --rm \
    --gpus all \
    --network host \
    --privileged \
    -v "$(pwd):/workspace/g1_deploy" \
    --name g1-real \
    g1-deploy-tensorrt10:latest \
    bash -c "source /opt/ros/humble/setup.bash && cd /workspace/g1_deploy && \
        ./target/release/g1_deploy_onnx_ref ${NETWORK_INTERFACE} ${DECODER_MODEL} ${MOTION_DATA} \
        --obs-config ${OBS_CONFIG} \
        --encoder-file ${ENCODER_MODEL} \
        --planner-file ${PLANNER_MODEL} \
        --input-type zmq_manager \
        --zmq-host ${ZMQ_HOST} \
        --zmq-port ${ZMQ_PORT} \
        --output-type ros2 \
        --verbose \
        --zmq-verbose \
        --enable-motion-recording \
        --enable-csv-logs"

# Note: For real robot:
#   --input-type zmq_manager → Receives VR commands from PICO via ZMQ
#   --output-type ros2        → Sends joint commands to robot via DDS (Unitree SDK2)
#   NO --disable-crc-check   → Safety CRC validation enabled
#   --privileged             → Required for real-time scheduling