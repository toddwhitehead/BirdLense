#!/bin/bash

# Exit on error
set -e

# CHECH PI CAM
echo -e "\nListing available cameras:"

# Detect which camera command is available
# rpicam-hello is used in newer versions of Raspberry Pi OS (Bookworm and later)
# libcamera-hello is used in older versions
if command -v rpicam-hello &> /dev/null; then
    camera_cmd="rpicam-hello"
elif command -v libcamera-hello &> /dev/null; then
    camera_cmd="libcamera-hello"
else
    echo "ERROR: Neither rpicam-hello nor libcamera-hello command found"
    echo "Please install libcamera-apps package"
    exit 1
fi

echo "Using camera command: $camera_cmd"
camera_list=$($camera_cmd --list-cameras 2>&1)
echo "$camera_list"
if ! echo "$camera_list" | grep -q "Available cameras"; then
    echo "ERROR: Could not list cameras"
    exit 1
fi
if echo "$camera_list" | grep -q "No cameras available"; then
    echo "ERROR: No cameras detected on the system"
    exit 1
fi
echo -e "Success: cameras detected\n"

# CHECK MICROPHONE
echo "Checking microphone... "
# First check if audio device exists
microphone_list=$(arecord -l 2>&1)
echo "$microphone_list"
if ! echo "$microphone_list" | grep -q 'card'; then
  echo "ERROR: No recording devices found"
  echo "Check if microphone is properly connected and detected by system"
  exit 1
fi
echo -e "Success: microphone detected\n"

# START INSTALLATION
echo "Starting BirdLense installation..."

# Update package lists
sudo apt-get update

# Install Docker
echo "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh

# Add current user to docker group
sudo usermod -aG docker $USER

# Configure Docker service delay to make sure all devices are ready
# echo "Configuring Docker service startup delay..."
# sudo mkdir -p /etc/systemd/system/docker.service.d/
# cat << EOF | sudo tee /etc/systemd/system/docker.service.d/override.conf
# [Unit]
# After=network-online.target firewalld.service containerd.service time-sync.target systemd-udev-settle.service
# Wants=network-online.target systemd-udev-settle.service
# RequiresMountsFor=%N
# [Service]
# ExecStartPre=/bin/sleep 15
# EOF

# Reload systemd configurations
# sudo systemctl daemon-reload

echo "---------------------"
echo "Installation completed!"
echo "You need to log out and log back in, or restart your terminal, to complete Docker installation and apply the group changes."
echo "After, run the following command to download and start the application: make start"