terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

variable "project_id" {
  type        = string
  description = "GCP Project ID"
}

variable "region" {
  type        = string
  default     = "us-central1"
}

variable "zone" {
  type        = string
  default     = "us-central1-a"
}

variable "use_spot" {
  type        = bool
  default     = true
  description = "Use Spot VM pricing for ~65-75% discount on 4x H100"
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

resource "google_storage_bucket" "checkpoints" {
  name                        = "${var.project_id}-qwen38-sre-rl-ckpts"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false
}

resource "google_service_account" "trainer" {
  account_id   = "qwen38-sre-rl-trainer"
  display_name = "Qwen3.8-27B SRE RLVR Trainer Service Account"
}

resource "google_storage_bucket_iam_member" "ckpt_writer" {
  bucket = google_storage_bucket.checkpoints.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.trainer.email}"
}

# a3-highgpu-4g: 4x NVIDIA H100 80GB HBM3, 104 vCPUs, 936 GB RAM, 3,000 GiB Local NVMe SSD
resource "google_compute_instance" "sre_rl_a3_4g" {
  name         = "qwen38-sre-rl-4xh100"
  machine_type = "a3-highgpu-4g"
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "projects/deeplearning-platform-release/global/images/family/pytorch-2-5-cu124-ubuntu-2204-nvidia-560"
      size  = 250
      type  = "pd-ssd"
    }
  }

  scheduling {
    on_host_maintenance          = "TERMINATE"
    automatic_restart            = var.use_spot ? false : true
    preemptible                  = var.use_spot
    provisioning_model           = var.use_spot ? "SPOT" : "STANDARD"
    instance_termination_action  = var.use_spot ? "STOP" : null
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata = {
    install-nvidia-driver = "True"
    gcs-checkpoint-bucket = google_storage_bucket.checkpoints.name
  }

  metadata_startup_script = file("${path.module}/startup_nvme_docker_cuda.sh")

  service_account {
    email  = google_service_account.trainer.email
    scopes = ["cloud-platform"]
  }
}

output "instance_name" {
  value = google_compute_instance.sre_rl_a3_4g.name
}

output "checkpoint_bucket" {
  value = "gs://${google_storage_bucket.checkpoints.name}"
}
