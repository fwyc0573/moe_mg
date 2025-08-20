#!/bin/bash

# Network Connectivity Debug Script for Multi-Node NCCL Testing
# This script helps diagnose network connectivity issues between nodes

set -e

# Configuration
MASTER_IP="10.232.194.226"
WORKER_IP="10.232.195.25"
TEST_PORT="29500"
ALTERNATIVE_PORTS=("29501" "29502" "29503")

echo "=========================================="
echo "Multi-Node Network Connectivity Debug"
echo "=========================================="
echo "Master IP: $MASTER_IP"
echo "Worker IP: $WORKER_IP"
echo "Test Port: $TEST_PORT"
echo ""

# Function to check if we're on master or worker node
get_node_role() {
    local current_ip=$(hostname -I | awk '{print $1}')
    if [[ "$current_ip" == "$MASTER_IP" ]]; then
        echo "master"
    elif [[ "$current_ip" == "$WORKER_IP" ]]; then
        echo "worker"
    else
        echo "unknown"
    fi
}

# Function to test basic network connectivity
test_basic_connectivity() {
    echo "1. Testing Basic Network Connectivity"
    echo "-------------------------------------"
    
    local node_role=$(get_node_role)
    local target_ip=""
    
    if [[ "$node_role" == "master" ]]; then
        target_ip="$WORKER_IP"
        echo "Testing connectivity from Master to Worker ($target_ip)"
    elif [[ "$node_role" == "worker" ]]; then
        target_ip="$MASTER_IP"
        echo "Testing connectivity from Worker to Master ($target_ip)"
    else
        echo "Unknown node role. Please run this script on master or worker node."
        return 1
    fi
    
    # Ping test
    echo "Ping test:"
    if ping -c 3 "$target_ip" > /dev/null 2>&1; then
        echo "✅ Ping successful to $target_ip"
    else
        echo "❌ Ping failed to $target_ip"
        echo "   Check network configuration and routing"
    fi
    
    echo ""
}

# Function to test port connectivity
test_port_connectivity() {
    echo "2. Testing Port Connectivity"
    echo "----------------------------"
    
    local node_role=$(get_node_role)
    
    if [[ "$node_role" == "master" ]]; then
        echo "Testing if Master can bind to port $TEST_PORT"
        
        # Test if port is already in use
        if netstat -tuln | grep ":$TEST_PORT " > /dev/null; then
            echo "❌ Port $TEST_PORT is already in use on Master"
            echo "   Processes using port $TEST_PORT:"
            netstat -tulnp | grep ":$TEST_PORT "
        else
            echo "✅ Port $TEST_PORT is available on Master"
        fi
        
        # Test alternative ports
        echo ""
        echo "Testing alternative ports:"
        for port in "${ALTERNATIVE_PORTS[@]}"; do
            if netstat -tuln | grep ":$port " > /dev/null; then
                echo "❌ Port $port is in use"
            else
                echo "✅ Port $port is available"
            fi
        done
        
    elif [[ "$node_role" == "worker" ]]; then
        echo "Testing if Worker can connect to Master port $TEST_PORT"
        
        # Test connection to master port
        if timeout 5 bash -c "echo >/dev/tcp/$MASTER_IP/$TEST_PORT" 2>/dev/null; then
            echo "✅ Can connect to Master port $TEST_PORT"
        else
            echo "❌ Cannot connect to Master port $TEST_PORT"
            echo "   This could indicate:"
            echo "   - Master node is not listening on this port"
            echo "   - Firewall is blocking the connection"
            echo "   - Network routing issues"
        fi
        
        # Test alternative ports
        echo ""
        echo "Testing alternative ports:"
        for port in "${ALTERNATIVE_PORTS[@]}"; do
            if timeout 2 bash -c "echo >/dev/tcp/$MASTER_IP/$port" 2>/dev/null; then
                echo "✅ Can connect to Master port $port"
            else
                echo "❌ Cannot connect to Master port $port"
            fi
        done
    fi
    
    echo ""
}

# Function to test firewall settings
test_firewall() {
    echo "3. Testing Firewall Configuration"
    echo "---------------------------------"
    
    # Check if firewall is active
    if command -v ufw >/dev/null 2>&1; then
        echo "UFW Firewall status:"
        sudo ufw status
    elif command -v firewall-cmd >/dev/null 2>&1; then
        echo "Firewalld status:"
        sudo firewall-cmd --state
        echo "Active zones:"
        sudo firewall-cmd --get-active-zones
    elif command -v iptables >/dev/null 2>&1; then
        echo "IPTables rules (first 10):"
        sudo iptables -L | head -10
    else
        echo "No common firewall tools found"
    fi
    
    echo ""
}

# Function to test InfiniBand connectivity
test_infiniband() {
    echo "4. Testing InfiniBand Connectivity"
    echo "----------------------------------"
    
    # Check if InfiniBand devices are available
    if command -v ibstat >/dev/null 2>&1; then
        echo "InfiniBand device status:"
        ibstat
        echo ""
        
        echo "InfiniBand port info:"
        ibportstate
        echo ""
    else
        echo "InfiniBand tools not found. Installing..."
        # Try to install InfiniBand tools
        if command -v yum >/dev/null 2>&1; then
            sudo yum install -y infiniband-diags
        elif command -v apt >/dev/null 2>&1; then
            sudo apt update && sudo apt install -y infiniband-diags
        else
            echo "Cannot install InfiniBand tools automatically"
        fi
    fi
    
    # Test RDMA connectivity if available
    if command -v ibping >/dev/null 2>&1; then
        local node_role=$(get_node_role)
        if [[ "$node_role" == "master" ]]; then
            echo "Starting ibping server on Master (run ibping client on Worker)"
            echo "Command for Worker: ibping -c $MASTER_IP"
        elif [[ "$node_role" == "worker" ]]; then
            echo "Testing ibping to Master"
            timeout 10 ibping -c "$MASTER_IP" || echo "ibping test failed or timed out"
        fi
    fi
    
    echo ""
}

# Function to provide recommendations
provide_recommendations() {
    echo "5. Recommendations"
    echo "------------------"
    
    local node_role=$(get_node_role)
    
    echo "Based on the test results, here are the recommendations:"
    echo ""
    
    echo "A. Use alternative port if 29500 is blocked:"
    echo "   Master: --master_port=29501"
    echo "   Worker: --master_port=29501"
    echo ""
    
    echo "B. Add timeout and retry mechanisms:"
    echo "   export NCCL_SOCKET_TIMEOUT=60000"
    echo "   export NCCL_CONNECT_TIMEOUT=60000"
    echo ""
    
    echo "C. Enable detailed NCCL debugging:"
    echo "   export NCCL_DEBUG=TRACE"
    echo "   export NCCL_DEBUG_SUBSYS=ALL"
    echo ""
    
    echo "D. Configure firewall (if needed):"
    if [[ "$node_role" == "master" ]]; then
        echo "   sudo ufw allow from $WORKER_IP to any port 29500:29510"
    elif [[ "$node_role" == "worker" ]]; then
        echo "   sudo ufw allow out to $MASTER_IP port 29500:29510"
    fi
    echo ""
    
    echo "E. Test with simple 2-process setup first:"
    echo "   Use smaller tensor sizes and fewer iterations"
    echo ""
}

# Main execution
main() {
    test_basic_connectivity
    test_port_connectivity
    test_firewall
    test_infiniband
    provide_recommendations
    
    echo "=========================================="
    echo "Network connectivity debug completed!"
    echo "=========================================="
}

# Run main function
main
