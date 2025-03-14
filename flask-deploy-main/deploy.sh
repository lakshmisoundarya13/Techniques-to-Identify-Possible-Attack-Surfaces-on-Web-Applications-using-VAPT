#!/bin/bash

echo "deleting old app"
sudo rm -rf /var/www/

echo "creating app folder"
sudo mkdir -p /var/www/flask_deploy 

echo "moving files to app folder"
sudo mv  * /var/www/flask_deploy

# Navigate to the app directory
cd /var/www/flask_deploy/

sudo apt-get update
echo "installing python and pip"
sudo apt-get install -y python3 python3-pip
sudo pip install flask_cors --break-system-packages
sudo pip install flask_login --break-system-packages

# Install application dependencies from requirements.txt
echo "Install application dependencies from requirements.txt"
sudo pip install -r requirements.txt ----break-system-packages
# sudo pkill gunicorn


sudo systemctl daemon-reload

sudo systemctl restart crud_app.service

echo "started crud_app.service"
sudo systemctl start crud_app.service

echo "enable crud_app.service"
sudo systemctl enable  crud_app.service

echo "status crud_app.service"
sudo systemctl status  crud_app.service


sudo systemctl restart nginx


# echo "starting gunicorn"
# sudo gunicorn --bind 0.0.0.0:8000 app:app
# echo "started gunicorn 🚀"







# # Update and install Nginx if not already installed
# if ! command -v nginx > /dev/null; then
#     echo "Installing Nginx"
#     sudo apt-get update
#     sudo apt-get install -y nginx
# fi

# # Configure Nginx to act as a reverse proxy if not already configured
# if [ ! -f /etc/nginx/sites-available/myapp ]; then
#     sudo rm -f /etc/nginx/sites-enabled/default
#     sudo bash -c 'cat > /etc/nginx/sites-available/myapp <<EOF
# server {
#     listen 80;
#     server_name _;

#     location / {
#         include proxy_params;
#         proxy_pass http://unix:/var/www/langchain-app/myapp.sock;
#     }
# }
# EOF'

#     sudo ln -s /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled
#     sudo systemctl restart nginx
# else
#     echo "Nginx reverse proxy configuration already exists."
# fi

# # Stop any existing Gunicorn process
# sudo pkill gunicorn
# # sudo rm -rf myapp.sock

# # # Start Gunicorn with the Flask application
# # # Replace 'server:app' with 'yourfile:app' if your Flask instance is named differently.
# # # gunicorn --workers 3 --bind 0.0.0.0:8000 server:app &
# echo "starting gunicorn"
# sudo gunicorn --workers 3 --bind unix:myapp.sock  server:app --user www-data --group www-data --daemon
# echo "started gunicorn 🚀"