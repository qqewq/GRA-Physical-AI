```bash
#!/bin/bash
# GRA Physical AI - Environment Setup Script
# ===========================================
#
# This script sets up the complete development environment for GRA Physical AI.
# It installs all dependencies, sets up virtual environments, and configures
# paths for various backends (PyBullet, Isaac Sim, ROS2, etc.)
#
# Usage: ./setup_env.sh [--help] [--dev] [--with-isaac] [--with-ros2] [--with-gr00t]
#
# Options:
#   --help          Show this help message
#   --dev           Install development dependencies (testing, linting)
#   --with-isaac    Install Isaac Sim dependencies (requires NVIDIA account)
#   --with-ros2     Install ROS 2 Humble dependencies
#   --with-gr00t    Install GR00T dependencies (requires NVIDIA access)
#   --with-all      Install all optional dependencies
#   --venv PATH     Specify virtual environment path (default: ./venv)
#   --python VER    Specify Python version (default: 3.10)

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
NC='\033[0m' # No Color

# Default values
VENV_PATH="./venv"
PYTHON_VERSION="3.10"
INSTALL_DEV=false
INSTALL_ISAAC=false
INSTALL_ROS2=false
INSTALL_GR00T=false
INSTALL_ALL=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ======================================================================
# Helper Functions
# ======================================================================

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
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

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 not found. Please install it first."
        return 1
    else
        print_success "$1 found: $($1 --version 2>&1 | head -n1)"
        return 0
    fi
}

confirm() {
    read -p "$1 (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# ======================================================================
# Parse Arguments
# ======================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            echo "GRA Physical AI - Environment Setup Script"
            echo ""
            echo "Usage: ./setup_env.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --help          Show this help message"
            echo "  --dev           Install development dependencies"
            echo "  --with-isaac    Install Isaac Sim dependencies"
            echo "  --with-ros2     Install ROS 2 Humble dependencies"
            echo "  --with-gr00t    Install GR00T dependencies"
            echo "  --with-all      Install all optional dependencies"
            echo "  --venv PATH     Virtual environment path (default: ./venv)"
            echo "  --python VER    Python version (default: 3.10)"
            echo ""
            exit 0
            ;;
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --with-isaac)
            INSTALL_ISAAC=true
            shift
            ;;
        --with-ros2)
            INSTALL_ROS2=true
            shift
            ;;
        --with-gr00t)
            INSTALL_GR00T=true
            shift
            ;;
        --with-all)
            INSTALL_DEV=true
            INSTALL_ISAAC=true
            INSTALL_ROS2=true
            INSTALL_GR00T=true
            INSTALL_ALL=true
            shift
            ;;
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ======================================================================
# System Checks
# ======================================================================

print_header "GRA Physical AI - Environment Setup"
print_info "Project root: $PROJECT_ROOT"
print_info "Python version: $PYTHON_VERSION"
print_info "Virtual environment: $VENV_PATH"
print_info "Options: dev=$INSTALL_DEV, isaac=$INSTALL_ISAAC, ros2=$INSTALL_ROS2, gr00t=$INSTALL_GR00T"

# Check Python version
print_header "Checking system requirements..."

if ! check_command python3; then
    print_error "Python 3 not found. Please install Python $PYTHON_VERSION"
    exit 1
fi

# Check Python version
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PY_VER" != "$PYTHON_VERSION"* ]]; then
    print_warning "Python version $PY_VER detected, but $PYTHON_VERSION recommended"
    if ! confirm "Continue with current version?"; then
        exit 1
    fi
else
    print_success "Python version $PY_VER"
fi

# Check pip
if ! check_command pip3; then
    print_error "pip3 not found. Please install pip"
    exit 1
fi

# Check CUDA (optional)
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    print_success "CUDA available: $CUDA_VERSION"
else
    print_warning "CUDA not found. GPU acceleration will not be available."
fi

# Check git
if ! check_command git; then
    print_error "git not found. Please install git"
    exit 1
fi

# ======================================================================
# Create Virtual Environment
# ======================================================================

print_header "Setting up virtual environment..."

# Check if venv already exists
if [ -d "$VENV_PATH" ]; then
    print_warning "Virtual environment already exists at $VENV_PATH"
    if confirm "Remove and recreate?"; then
        rm -rf "$VENV_PATH"
    else
        print_info "Using existing virtual environment"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    print_info "Creating virtual environment at $VENV_PATH"
    python3 -m venv "$VENV_PATH"
    print_success "Virtual environment created"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate" || source "$VENV_PATH/Scripts/activate"  # Windows fallback

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# ======================================================================
# Install Core Dependencies
# ======================================================================

print_header "Installing core dependencies..."

# Core scientific libraries
print_info "Installing scientific libraries..."
pip install numpy==1.24.3
pip install scipy==1.10.1
pip install pandas==2.0.3
pip install matplotlib==3.7.2
pip install seaborn==0.12.2

# PyTorch
print_info "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    # CUDA version
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
else
    # CPU version
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
fi

# Machine learning libraries
print_info "Installing ML libraries..."
pip install scikit-learn==1.3.0
pip install gymnasium==0.28.1
pip install stable-baselines3==2.1.0
pip install transformers==4.31.0

# Visualization
print_info "Installing visualization libraries..."
pip install plotly==5.15.0
pip install ipywidgets==8.0.7
pip install jupyterlab==4.0.3

# Utilities
print_info "Installing utilities..."
pip install pyyaml==6.0
pip install tqdm==4.65.0
pip install h5py==3.9.0
pip install psutil==5.9.5
pip install colorama==0.4.6

# ======================================================================
# Install Development Dependencies
# ======================================================================

if [ "$INSTALL_DEV" = true ] || [ "$INSTALL_ALL" = true ]; then
    print_header "Installing development dependencies..."
    
    # Testing
    pip install pytest==7.4.0
    pip install pytest-cov==4.1.0
    pip install pytest-xdist==3.3.1
    pip install pytest-timeout==2.1.0
    
    # Linting and formatting
    pip install black==23.7.0
    pip install flake8==6.1.0
    pip install pylint==2.17.4
    pip install mypy==1.4.1
    pip install isort==5.12.0
    
    # Documentation
    pip install sphinx==7.1.2
    pip install sphinx-rtd-theme==1.2.2
    
    # Pre-commit hooks
    pip install pre-commit==3.3.3
    
    print_success "Development dependencies installed"
fi

# ======================================================================
# Install PyBullet
# ======================================================================

print_header "Installing PyBullet..."
pip install pybullet==3.2.5
print_success "PyBullet installed"

# Download PyBullet assets
print_info "Downloading PyBullet assets..."
python -c "import pybullet_data; print(f'PyBullet data path: {pybullet_data.getDataPath()}')"

# ======================================================================
# Install MuJoCo
# ======================================================================

print_header "Installing MuJoCo..."
pip install mujoco==2.3.7
pip install dm-control==1.0.11
print_success "MuJoCo installed"

# ======================================================================
# Install ROS 2 Dependencies
# ======================================================================

if [ "$INSTALL_ROS2" = true ] || [ "$INSTALL_ALL" = true ]; then
    print_header "Installing ROS 2 dependencies..."
    
    # Check if ROS 2 is installed
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        print_success "ROS 2 Humble found"
        
        # Install ROS 2 Python packages
        pip install rclpy==0.10.1
        pip install rosbag2_py==0.15.2
        pip install tf2_ros==0.18.1
        pip install tf2_geometry_msgs==0.18.1
        pip install cv_bridge==3.2.1
        
        print_success "ROS 2 Python packages installed"
    else
        print_warning "ROS 2 Humble not found in /opt/ros/humble"
        print_info "Skipping ROS 2 package installation"
        print_info "To install ROS 2, follow: https://docs.ros.org/en/humble/Installation.html"
    fi
fi

# ======================================================================
# Install Isaac Sim Dependencies
# ======================================================================

if [ "$INSTALL_ISAAC" = true ] || [ "$INSTALL_ALL" = true ]; then
    print_header "Installing Isaac Sim dependencies..."
    
    # Check if ISAAC_PATH is set
    if [ -z "${ISAAC_PATH:-}" ]; then
        print_warning "ISAAC_PATH environment variable not set"
        print_info "Please set ISAAC_PATH to your Isaac Sim installation"
        print_info "Example: export ISAAC_PATH=~/isaac_sim-2023.1.1"
        
        # Try common paths
        if [ -d "$HOME/isaac_sim-2023.1.1" ]; then
            export ISAAC_PATH="$HOME/isaac_sim-2023.1.1"
            print_success "Found Isaac Sim at $ISAAC_PATH"
        elif [ -d "$HOME/isaac_sim" ]; then
            export ISAAC_PATH="$HOME/isaac_sim"
            print_success "Found Isaac Sim at $ISAAC_PATH"
        fi
    fi
    
    if [ -n "${ISAAC_PATH:-}" ]; then
        print_info "Installing Isaac Sim Python bindings..."
        
        # Add Isaac Sim to PYTHONPATH
        export PYTHONPATH="${ISAAC_PATH}/python:${PYTHONPATH:-}"
        
        # Install Isaac Sim dependencies
        pip install carb
        pip install omni-kit
        
        print_success "Isaac Sim dependencies installed"
        print_info "Note: You need to accept NVIDIA's license agreement"
    else
        print_warning "Isaac Sim not found. Skipping Isaac Sim installation."
        print_info "Download from: https://developer.nvidia.com/isaac-sim"
    fi
fi

# ======================================================================
# Install GR00T Dependencies
# ======================================================================

if [ "$INSTALL_GR00T" = true ] || [ "$INSTALL_ALL" = true ]; then
    print_header "Installing GR00T dependencies..."
    
    # Check if GR00T_MODEL_PATH is set
    if [ -z "${GR00T_MODEL_PATH:-}" ]; then
        print_warning "GR00T_MODEL_PATH environment variable not set"
        print_info "Please set GR00T_MODEL_PATH to your GR00T model directory"
    else
        if [ -d "$GR00T_MODEL_PATH" ]; then
            print_success "GR00T model found at $GR00T_MODEL_PATH"
            pip install gr00t-toolkit  # Placeholder - actual package name may differ
        else
            print_warning "GR00T model not found at $GR00T_MODEL_PATH"
        fi
    fi
    
    print_info "GR00T requires access from NVIDIA. Please visit: https://developer.nvidia.com/gr00t"
fi

# ======================================================================
# Install GRA Package in Development Mode
# ======================================================================

print_header "Installing GRA Physical AI package..."

# Install the package in development mode
cd "$PROJECT_ROOT"
pip install -e .

print_success "GRA Physical AI package installed in development mode"

# ======================================================================
# Setup Configuration Files
# ======================================================================

print_header "Setting up configuration files..."

# Create config directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/configs"

# Create default configuration if it doesn't exist
if [ ! -f "$PROJECT_ROOT/configs/default.yaml" ]; then
    cat > "$PROJECT_ROOT/configs/default.yaml" << EOF
# GRA Physical AI - Default Configuration
experiment_id: "default"
description: "Default configuration"
environment:
  type: "pybullet"
  name: "cartpole"
  max_steps: 1000
  gui: false
agent:
  type: "mlp"
  hidden_dims: [256, 256]
  learning_rate: 0.0003
gra:
  use_g0: true
  use_g1: false
  use_g2: false
  use_g3: false
  num_joints: 2
training:
  total_episodes: 100
  max_steps_per_episode: 500
  log_dir: "./logs"
  checkpoint_dir: "./checkpoints"
EOF
    print_success "Created default configuration at configs/default.yaml"
fi

# ======================================================================
# Create Activation Script
# ======================================================================

print_header "Creating activation script..."

cat > "$PROJECT_ROOT/activate_env.sh" << EOF
#!/bin/bash
# Activate GRA Physical AI environment

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Set environment variables
export GRA_ROOT="$PROJECT_ROOT"
export PYTHONPATH="\$GRA_ROOT:\$PYTHONPATH"

# Optional: Set paths for specific backends
# export ISAAC_PATH="~/isaac_sim-2023.1.1"
# export GR00T_MODEL_PATH="~/gr00t_models"

echo "GRA Physical AI environment activated"
echo "Python: \$(which python)"
echo "GRA_ROOT: \$GRA_ROOT"
EOF

chmod +x "$PROJECT_ROOT/activate_env.sh"
print_success "Created activation script at activate_env.sh"

# ======================================================================
# Run Tests
# ======================================================================

if [ "$INSTALL_DEV" = true ] || [ "$INSTALL_ALL" = true ]; then
    print_header "Running core tests..."
    
    cd "$PROJECT_ROOT"
    python -m pytest tests/test_core.py -v
    
    if [ $? -eq 0 ]; then
        print_success "Core tests passed"
    else
        print_warning "Some core tests failed"
    fi
fi

# ======================================================================
# Final Steps
# ======================================================================

print_header "Setup Complete!"

print_success "GRA Physical AI environment has been set up successfully!"
print_info ""
print_info "Next steps:"
print_info "  1. Activate the environment:"
print_info "     source $PROJECT_ROOT/activate_env.sh"
print_info ""
print_info "  2. Run a quick test:"
print_info "     cd $PROJECT_ROOT/examples"
print_info "     python mobile_robot/run.py"
print_info ""
print_info "  3. Explore the documentation:"
print_info "     cd $PROJECT_ROOT/docs"
print_info "     # View the markdown files"
print_info ""
print_info "  4. (Optional) Set up environment variables in activate_env.sh"
print_info ""

# Print summary
echo -e "\n${GREEN}Installation Summary:${NC}"
echo -e "  Python: $PY_VER"
echo -e "  Virtual env: $VENV_PATH"
echo -e "  Core packages: ✓"
echo -e "  Development: $INSTALL_DEV"
echo -e "  ROS 2: $INSTALL_ROS2"
echo -e "  Isaac Sim: $INSTALL_ISAAC"
echo -e "  GR00T: $INSTALL_GR00T"

# Check for any missing optional components
echo -e "\n${YELLOW}Optional Components:${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "  - CUDA: Not found (CPU only)"
fi

if [ ! -d "/opt/ros/humble" ]; then
    echo -e "  - ROS 2: Not found"
fi

if [ -z "${ISAAC_PATH:-}" ]; then
    echo -e "  - Isaac Sim: Not configured"
fi

echo -e "\n${GREEN}Happy experimenting with GRA Physical AI!${NC}\n"
```