```bash
#!/bin/bash
# GRA Physical AI - Demo Runner Script
# ======================================
#
# This script runs various demonstrations of GRA Physical AI capabilities.
# It provides easy access to pre-configured demos with proper environment setup.
#
# Usage: ./run_demo.sh [demo_name] [options]
#
# Available demos:
#   mobile_robot          - Two-wheeled robot with G0/G1 layers
#   humanoid_safety       - Humanoid robot with safety layer (G2)
#   multi_agent           - Multiple robots with coordination
#   cartpole_basic        - Simple CartPole with G0 only
#   gr00t_mimic           - GR00T-Mimic fine-tuning demo
#   isaac_integration     - Isaac Sim integration demo
#   foam_analysis         - Analyze foam from logs
#   list                  - List all available demos
#
# Options:
#   --help                Show this help message
#   --gui                 Enable GUI/visualization
#   --headless            Run without GUI (for servers)
#   --config FILE         Use custom config file
#   --episodes N          Number of episodes (for training demos)
#   --steps N             Number of steps (for evaluation demos)
#   --device DEVICE       Device to use (cpu/cuda)
#   --log-dir DIR         Directory for logs
#   --verbose             Enable verbose output

set -e  # Exit on error
set -u  # Exit on undefined variable

# ======================================================================
# Configuration
# ======================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default options
DEMO_NAME=""
GUI=true
HEADLESS=false
CONFIG_FILE=""
EPISODES=10
STEPS=100
DEVICE="auto"
LOG_DIR="$PROJECT_ROOT/logs"
VERBOSE=false

# ======================================================================
# Helper Functions
# ======================================================================

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

print_demo() {
    echo -e "${CYAN}  $1${NC}"
    echo -e "     $2"
}

check_environment() {
    # Check if virtual environment is activated
    if [ -z "${VIRTUAL_ENV:-}" ]; then
        # Try to activate from common locations
        if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
            print_info "Activating virtual environment from $PROJECT_ROOT/venv"
            source "$PROJECT_ROOT/venv/bin/activate"
        elif [ -f "$PROJECT_ROOT/activate_env.sh" ]; then
            print_info "Running activation script"
            source "$PROJECT_ROOT/activate_env.sh"
        else
            print_warning "No virtual environment found. Using system Python."
        fi
    else
        print_success "Using virtual environment: $VIRTUAL_ENV"
    fi
    
    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found"
        exit 1
    fi
    
    PY_VERSION=$(python --version 2>&1)
    print_success "Python: $PY_VERSION"
}

check_demo_available() {
    local demo=$1
    
    case $demo in
        mobile_robot|humanoid_safety|multi_agent|cartpole_basic|gr00t_mimic|isaac_integration|foam_analysis|list)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

list_demos() {
    print_header "Available Demos"
    print_demo "mobile_robot" "Two-wheeled robot with G0/G1 layers"
    print_demo "humanoid_safety" "Humanoid robot with safety layer (G2)"
    print_demo "multi_agent" "Multiple robots with coordination"
    print_demo "cartpole_basic" "Simple CartPole with G0 only"
    print_demo "gr00t_mimic" "GR00T-Mimic fine-tuning demo (requires GR00T)"
    print_demo "isaac_integration" "Isaac Sim integration demo (requires Isaac Sim)"
    print_demo "foam_analysis" "Analyze foam from logs"
    echo ""
}

show_help() {
    cat << EOF
GRA Physical AI - Demo Runner Script

Usage: ./run_demo.sh [demo_name] [options]

Available demos:
  mobile_robot          - Two-wheeled robot with G0/G1 layers
  humanoid_safety       - Humanoid robot with safety layer (G2)
  multi_agent           - Multiple robots with coordination
  cartpole_basic        - Simple CartPole with G0 only
  gr00t_mimic           - GR00T-Mimic fine-tuning demo (requires GR00T)
  isaac_integration     - Isaac Sim integration demo (requires Isaac Sim)
  foam_analysis         - Analyze foam from logs
  list                  - List all available demos

Options:
  --help                Show this help message
  --gui                 Enable GUI/visualization (default: true)
  --headless            Run without GUI (for servers)
  --config FILE         Use custom config file
  --episodes N          Number of episodes (for training demos)
  --steps N             Number of steps (for evaluation demos)
  --device DEVICE       Device to use (cpu/cuda/auto)
  --log-dir DIR         Directory for logs
  --verbose             Enable verbose output

Examples:
  ./run_demo.sh mobile_robot --gui
  ./run_demo.sh humanoid_safety --headless --episodes 50
  ./run_demo.sh foam_analysis --log-dir ./logs/my_experiment
  ./run_demo.sh list
EOF
}

# ======================================================================
# Parse Arguments
# ======================================================================

# First argument is demo name (if it doesn't start with --)
if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
    DEMO_NAME="$1"
    shift
fi

# Parse remaining options
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --gui)
            GUI=true
            HEADLESS=false
            shift
            ;;
        --headless)
            GUI=false
            HEADLESS=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# If no demo specified, show help
if [ -z "$DEMO_NAME" ]; then
    show_help
    exit 0
fi

# Check if demo exists
if ! check_demo_available "$DEMO_NAME"; then
    print_error "Unknown demo: $DEMO_NAME"
    list_demos
    exit 1
fi

# Handle 'list' command
if [ "$DEMO_NAME" = "list" ]; then
    list_demos
    exit 0
fi

# ======================================================================
# Environment Setup
# ======================================================================

print_header "GRA Physical AI - Demo Runner"
print_info "Demo: $DEMO_NAME"
print_info "Options: GUI=$GUI, Episodes=$EPISODES, Steps=$STEPS, Device=$DEVICE"

# Check environment
check_environment

# Set device
if [ "$DEVICE" = "auto" ]; then
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        DEVICE="cuda"
        print_success "CUDA available, using GPU"
    else
        DEVICE="cpu"
        print_info "CUDA not available, using CPU"
    fi
fi

# Create log directory
mkdir -p "$LOG_DIR"
print_success "Log directory: $LOG_DIR"

# ======================================================================
# Demo-Specific Setup
# ======================================================================

case $DEMO_NAME in
    mobile_robot)
        print_header "Mobile Robot Demo (G0/G1 Layers)"
        
        # Build command
        CMD="python $PROJECT_ROOT/examples/mobile_robot/run.py"
        CMD="$CMD --episodes $EPISODES"
        CMD="$CMD --log-dir $LOG_DIR/mobile_robot"
        CMD="$CMD --device $DEVICE"
        
        if [ "$GUI" = true ]; then
            CMD="$CMD --gui"
        else
            CMD="$CMD --no-gui"
        fi
        
        if [ -n "$CONFIG_FILE" ]; then
            CMD="$CMD --config $CONFIG_FILE"
        fi
        
        if [ "$VERBOSE" = true ]; then
            CMD="$CMD --verbose"
        fi
        
        print_info "Command: $CMD"
        echo ""
        
        # Run demo
        eval $CMD
        ;;
        
    humanoid_safety)
        print_header "Humanoid Safety Demo (G2 Layer)"
        
        # Build command
        CMD="python $PROJECT_ROOT/examples/humanoid_safety/run.py"
        CMD="$CMD --episodes $EPISODES"
        CMD="$CMD --log-dir $LOG_DIR/humanoid_safety"
        CMD="$CMD --device $DEVICE"
        
        if [ "$GUI" = true ]; then
            CMD="$CMD --gui"
        else
            CMD="$CMD --no-gui"
        fi
        
        if [ -n "$CONFIG_FILE" ]; then
            CMD="$CMD --config $CONFIG_FILE"
        fi
        
        if [ "$VERBOSE" = true ]; then
            CMD="$CMD --verbose"
        fi
        
        print_info "Command: $CMD"
        echo ""
        
        # Run demo
        eval $CMD
        ;;
        
    multi_agent)
        print_header "Multi-Agent Coordination Demo"
        
        # Build command
        CMD="python $PROJECT_ROOT/examples/multi_agent/run.py"
        CMD="$CMD --episodes $EPISODES"
        CMD="$CMD --log-dir $LOG_DIR/multi_agent"
        CMD="$CMD --device $DEVICE"
        
        if [ "$GUI" = true ]; then
            CMD="$CMD --gui"
        else
            CMD="$CMD --no-gui"
        fi
        
        if [ -n "$CONFIG_FILE" ]; then
            CMD="$CMD --config $CONFIG_FILE"
        fi
        
        if [ "$VERBOSE" = true ]; then
            CMD="$CMD --verbose"
        fi
        
        print_info "Command: $CMD"
        echo ""
        
        # Run demo
        eval $CMD
        ;;
        
    cartpole_basic)
        print_header "CartPole Basic Demo (G0 Only)"
        
        # Build command
        CMD="python $PROJECT_ROOT/examples/cartpole/run.py"
        CMD="$CMD --episodes $EPISODES"
        CMD="$CMD --log-dir $LOG_DIR/cartpole"
        CMD="$CMD --device $DEVICE"
        
        if [ "$GUI" = true ]; then
            CMD="$CMD --gui"
        else
            CMD="$CMD --no-gui"
        fi
        
        if [ -n "$CONFIG_FILE" ]; then
            CMD="$CMD --config $CONFIG_FILE"
        fi
        
        if [ "$VERBOSE" = true ]; then
            CMD="$CMD --verbose"
        fi
        
        print_info "Command: $CMD"
        echo ""
        
        # Run demo
        eval $CMD
        ;;
        
    gr00t_mimic)
        print_header "GR00T-Mimic Fine-tuning Demo"
        
        # Check if GR00T is available
        if [ -z "${GR00T_MODEL_PATH:-}" ]; then
            print_warning "GR00T_MODEL_PATH not set"
            print_info "This demo requires GR00T models from NVIDIA"
            
            if ! confirm "Continue without GR00T (using mock model)?"; then
                exit 1
            fi
        fi
        
        # Build command
        CMD="python $PROJECT_ROOT/examples/gr00t_mimic/finetune.py"
        CMD="$CMD --config $PROJECT_ROOT/configs/gr00t_mimic.yaml"
        CMD="$CMD --steps $EPISODES"
        CMD="$CMD --log-dir $LOG_DIR/gr00t_mimic"
        CMD="$CMD --device $DEVICE"
        
        if [ -n "$CONFIG_FILE" ]; then
            CMD="$CMD --config $CONFIG_FILE"
        fi
        
        if [ "$VERBOSE" = true ]; then
            CMD="$CMD --verbose"
        fi
        
        print_info "Command: $CMD"
        echo ""
        
        # Run demo
        eval $CMD
        ;;
        
    isaac_integration)
        print_header "Isaac Sim Integration Demo"
        
        # Check if Isaac Sim is available
        if [ -z "${ISAAC_PATH:-}" ]; then
            print_error "ISAAC_PATH environment variable not set"
            print_info "Please set ISAAC_PATH to your Isaac Sim installation"
            exit 1
        fi
        
        # Build command
        CMD="python $PROJECT_ROOT/examples/isaac_integration/run.py"
        CMD="$CMD --episodes $EPISODES"
        CMD="$CMD --log-dir $LOG_DIR/isaac"
        CMD="$CMD --device $DEVICE"
        
        if [ "$GUI" = true ]; then
            CMD="$CMD --gui"
        else
            CMD="$CMD --headless"
        fi
        
        if [ -n "$CONFIG_FILE" ]; then
            CMD="$CMD --config $CONFIG_FILE"
        fi
        
        if [ "$VERBOSE" = true ]; then
            CMD="$CMD --verbose"
        fi
        
        print_info "Command: $CMD"
        echo ""
        
        # Set Isaac Sim environment
        export ISAAC_SIM_PATH="$ISAAC_PATH"
        export PYTHONPATH="${ISAAC_PATH}/python:${PYTHONPATH:-}"
        
        # Run demo
        eval $CMD
        ;;
        
    foam_analysis)
        print_header "Foam Analysis Demo"
        
        # Build command
        CMD="python $PROJECT_ROOT/examples/foam_analysis/analyze.py"
        CMD="$CMD --log-dir $LOG_DIR"
        
        if [ -n "$CONFIG_FILE" ]; then
            CMD="$CMD --config $CONFIG_FILE"
        fi
        
        if [ "$VERBOSE" = true ]; then
            CMD="$CMD --verbose"
        fi
        
        print_info "Command: $CMD"
        echo ""
        
        # Run demo
        eval $CMD
        ;;
esac

# ======================================================================
# Post-Run Processing
# ======================================================================

# Check exit status
if [ $? -eq 0 ]; then
    print_success "Demo completed successfully!"
    
    # Show log location
    print_info "Logs saved to: $LOG_DIR/$DEMO_NAME"
    
    # Offer to visualize results
    if [ "$DEMO_NAME" != "foam_analysis" ] && [ "$GUI" = false ]; then
        echo ""
        print_info "To analyze results, run:"
        print_info "  ./run_demo.sh foam_analysis --log-dir $LOG_DIR/$DEMO_NAME"
    fi
else
    print_error "Demo failed with exit code $?"
    exit 1
fi

echo ""
print_success "Done!"
```