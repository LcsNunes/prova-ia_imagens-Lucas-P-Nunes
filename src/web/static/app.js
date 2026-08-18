const input = document.querySelector("#image-input");
const dropzone = document.querySelector("#dropzone");
const analyzeButton = document.querySelector("#analyze-button");
const message = document.querySelector("#message");
const roadToggle = document.querySelector("#road-toggle");
const roadCard = document.querySelector("#road-card");

const originalImage = document.querySelector("#original-image");
const detectionsImage = document.querySelector("#detections-image");
const combinedImage = document.querySelector("#combined-image");
const roadsImage = document.querySelector("#roads-image");

function updateMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function selectedFile() {
  return input.files && input.files[0];
}

input.addEventListener("change", () => {
  const file = selectedFile();
  analyzeButton.disabled = !file;
  dropzone.classList.toggle("has-file", Boolean(file));
  if (!file) return;
  originalImage.src = URL.createObjectURL(file);
  updateMessage(`${file.name} pronta para analise.`);
});

analyzeButton.addEventListener("click", async () => {
  const file = selectedFile();
  if (!file) return;

  analyzeButton.disabled = true;
  roadToggle.disabled = true;
  updateMessage("Detectando veiculos e segmentando a malha viaria...");
  const formData = new FormData();
  formData.append("image", file);

  try {
    const response = await fetch("/api/analyses", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail?.message || "Nao foi possivel analisar a imagem.");
    }
    document.querySelector("#vehicle-count").textContent = payload.vehicle_count;
    document.querySelector("#model-name").textContent = payload.model_name;
    document.querySelector("#inference-time").textContent = `${payload.inference_ms.toFixed(0)} ms`;
    document.querySelector("#sahi-status").textContent = payload.sahi_enabled ? "Ligado" : "Desligado";
    detectionsImage.src = payload.detections_image;
    combinedImage.src = payload.combined_image;
    roadsImage.src = payload.roads_image;
    roadToggle.disabled = false;
    updateMessage(
      "Analise concluida: veiculos " +
        payload.vehicle_inference_ms.toFixed(0) +
        " ms | vias " +
        payload.road_inference_ms.toFixed(0) +
        " ms. Caixas ciano, vias laranja.",
    );
  } catch (error) {
    updateMessage(error.message, true);
  } finally {
    analyzeButton.disabled = false;
  }
});

roadToggle.addEventListener("click", () => {
  const hidden = roadCard.classList.toggle("is-hidden");
  roadToggle.textContent = hidden ? "Mostrar apenas as vias" : "Ocultar camada de vias";
});

fetch("/api/health")
  .then((response) => response.json())
  .then((payload) => {
    const status = document.querySelector(".status");
    const label = document.querySelector("#api-status");
    status.classList.toggle("ready", payload.status === "ok");
    label.textContent = payload.model_loaded ? "API e modelo prontos" : "API pronta";
  })
  .catch(() => updateMessage("Nao foi possivel verificar a API local.", true));
