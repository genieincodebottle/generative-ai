# Use an official Node runtime as the parent image
FROM node:14

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the current directory contents into the container
COPY . .

# Build the app
RUN npm run build

# Install serve to run the application
RUN npm install -g serve

# Specify the command to run on container start
CMD ["serve", "-s", "build", "-l", "3000"]