terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "kubernetes" {
  config_path    = "~/.kube/config"
  config_context = "minikube"
}

resource "kubernetes_namespace" "fraud_detection" {
  metadata {
    name = "fraud-detection-ns"
    labels = {
      project    = "fraud-detection-mlops"
      student    = "kaab-abdullah-malik"
      sap_id     = "70148009"
      managed_by = "terraform"
    }
  }
}

resource "kubernetes_deployment" "fraud_api" {
  metadata {
    name      = "fraud-detection-terraform"
    namespace = kubernetes_namespace.fraud_detection.metadata[0].name
    labels = {
      app = "fraud-detection"
    }
  }

  spec {
    replicas = 3

    selector {
      match_labels = {
        app = "fraud-detection"
      }
    }

    template {
      metadata {
        labels = {
          app = "fraud-detection"
        }
      }

      spec {
        container {
          name              = "fraud-api"
          image             = "fraud-detection-api:v1"
          image_pull_policy = "Never"

          port {
            container_port = 8000
          }

          resources {
            requests = {
              memory = "256Mi"
              cpu    = "250m"
            }
            limits = {
              memory = "512Mi"
              cpu    = "500m"
            }
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "fraud_service" {
  metadata {
    name      = "fraud-detection-tf-service"
    namespace = kubernetes_namespace.fraud_detection.metadata[0].name
  }

  spec {
    selector = {
      app = "fraud-detection"
    }

    port {
      port        = 80
      target_port = 8000
    }

    type = "NodePort"
  }
}

output "namespace" {
  value = kubernetes_namespace.fraud_detection.metadata[0].name
}
