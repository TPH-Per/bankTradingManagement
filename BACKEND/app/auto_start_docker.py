"""
Auto-start Docker services for Bank Trading Management System
This script is called during backend startup to ensure all required services are running.
"""

import subprocess
import time
import os
import sys
import socket
from pathlib import Path

def check_port_in_use(host: str, port: int) -> bool:
    """Check if a port is already in use"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception:
        return False

def is_docker_running() -> bool:
    """Check if Docker daemon is running"""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Check if both client and server are shown (server = daemon running)
        return "Server:" in result.stdout and result.returncode == 0
    except Exception:
        return False

def is_container_running(container_name: str) -> bool:
    """Check if a Docker container is running"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return container_name in result.stdout
    except Exception:
        return False

def start_docker_services(project_dir: Path, use_cassandra_docker: bool = False):
    """
    Start all required Docker services.
    
    Args:
        project_dir: Root directory of the project (where docker-compose.yml is)
        use_cassandra_docker: If True, start Cassandra in Docker (default: False, use Windows native)
    """
    print("\n" + "="*70)
    print("🐳 Auto-starting Docker Services")
    print("="*70)
    
    # Change to project directory
    original_dir = os.getcwd()
    os.chdir(project_dir)
    
    try:
        # Step 1: Check if Docker is running
        print("\n[1/4] Checking Docker daemon...")
        if not is_docker_running():
            print("   ⚠️  Docker is not running!")
            print("   Please start Docker Desktop and run the backend again.")
            print("   Services will run in fallback mode (without Docker).")
            return False
        print("   ✅ Docker is running")
        
        # Step 2: Check which services need to be started
        print("\n[2/4] Checking service status...")
        
        services_to_start = []
        
        # Cassandra (optional - prefer Windows native)
        if use_cassandra_docker:
            if not check_port_in_use("localhost", 9042):
                if not is_container_running("bt-cassandra"):
                    services_to_start.append("cassandra")
                    print("   📦 Will start: Cassandra (Docker)")
                else:
                    print("   ✅ Cassandra container already running")
            else:
                print("   ℹ️  Cassandra already running (port 9042 - likely Windows native)")
        else:
            print("   ℹ️  Using Windows native Cassandra (skipping Docker Cassandra)")
        
        # Redis
        if not is_container_running("bt-redis"):
            services_to_start.append("redis")
            print("   📦 Will start: Redis")
        else:
            print("   ✅ Redis already running")
        
        # HDFS
        if not is_container_running("bt-hdfs-namenode"):
            services_to_start.append("hdfs-namenode")
            services_to_start.append("hdfs-datanode")
            print("   📦 Will start: HDFS (NameNode + DataNode)")
        else:
            print("   ✅ HDFS already running")
        
        # Step 3: Start services if needed
        if services_to_start:
            print(f"\n[3/4] Starting {len(services_to_start)} service(s)...")
            
            # Use docker-compose to start specific services
            cmd = ["docker-compose", "up", "-d"] + services_to_start
            
            print(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout
            )
            
            if result.returncode == 0:
                print("   ✅ Services started successfully!")
                
                # Wait a bit for services to initialize
                print("\n   ⏳ Waiting for services to initialize (15 seconds)...")
                time.sleep(15)
            else:
                print(f"   ❌ Failed to start services!")
                print(f"   Error: {result.stderr}")
                return False
        else:
            print("\n[3/4] All required services already running")
        
        # Step 4: Verify services
        print("\n[4/4] Verifying services...")
        
        all_healthy = True
        
        # Check Redis
        if check_port_in_use("localhost", 6379):
            print("   ✅ Redis is accessible (port 6379)")
        else:
            print("   ⚠️  Redis port 6379 not accessible")
            all_healthy = False
        
        # Check HDFS NameNode
        if check_port_in_use("localhost", 9870):
            print("   ✅ HDFS NameNode is accessible (port 9870)")
        else:
            print("   ⚠️  HDFS NameNode port 9870 not accessible (may still be initializing)")
        
        # Check Cassandra (if using Docker)
        if use_cassandra_docker:
            if check_port_in_use("localhost", 9042):
                print("   ✅ Cassandra is accessible (port 9042)")
            else:
                print("   ⚠️  Cassandra port 9042 not accessible")
                all_healthy = False
        
        print("\n" + "="*70)
        if all_healthy:
            print("🎉 All Docker services are ready!")
        else:
            print("⚠️  Some services may still be initializing...")
            print("Backend will continue and retry connections automatically.")
        print("="*70 + "\n")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("   ❌ Timeout starting Docker services!")
        return False
    except Exception as e:
        print(f"   ❌ Error starting Docker services: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Return to original directory
        os.chdir(original_dir)

def main():
    """Main entry point for standalone execution"""
    # Get project directory (parent of BACKEND)
    backend_dir = Path(__file__).parent.parent
    project_dir = backend_dir.parent
    
    print(f"Project directory: {project_dir}")
    
    # Start services (use Windows Cassandra by default)
    success = start_docker_services(
        project_dir=project_dir,
        use_cassandra_docker=False  # Set to True to use Docker Cassandra
    )
    
    if success:
        print("\n✅ Docker services auto-start completed!")
        sys.exit(0)
    else:
        print("\n⚠️  Docker services auto-start had issues.")
        print("Backend will continue in fallback mode.")
        sys.exit(1)

if __name__ == "__main__":
    main()
