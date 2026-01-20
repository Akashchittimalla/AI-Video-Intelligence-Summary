terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0.1"
    }
  }
}

provider "docker" {}

resource "docker_image" "video_ai" {
  name         = "video-ai-intelligence:latest"
  keep_locally = true
}

resource "docker_container" "video_ai_container" {
  image = docker_image.video_ai.image_id
  name  = "video_ai_instance"
  
  # Connects the container to your local Ollama instance
  host {
    host = "host.docker.internal"
    ip   = "127.0.0.1"
  }
}