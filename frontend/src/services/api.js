import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

/**
 * Sends image file to FastAPI backend for prediction.
 * @param {File} file - Image file to predict
 * @returns {Promise<{label: string, confidence: number}>}
 */
export async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await apiClient.post("/predict", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return response.data;
}

/**
 * Pings backend health endpoint.
 * @returns {Promise<{status: string}>}
 */
export async function checkHealth() {
  const response = await apiClient.get("/health");
  return response.data;
}

export default {
  predictImage,
  checkHealth,
};
