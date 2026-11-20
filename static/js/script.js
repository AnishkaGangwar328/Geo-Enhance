document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const browseBtn = document.getElementById("browseBtn");
    const enhanceBtn = document.getElementById("enhanceBtn");
    const selectedFile = document.getElementById("selectedFile");
    const fileName = document.getElementById("fileName");
    const processingState = document.getElementById("processingState");
    const errorMessage = document.getElementById("errorMessage");

    const resultsSection = document.getElementById("resultsSection");
    const lrImage = document.getElementById("lrImage");
    const srImage = document.getElementById("srImage");
    const downloadBtn = document.getElementById("downloadBtn");
    const resultFilename = document.getElementById("resultFilename");

    let currentFile = null;

    // Preview-only effect. Does not modify uploaded GeoTIFF.
    const PREVIEW_BLUR_PX = 0.8;
    const PREVIEW_NOISE_STRENGTH = 8;

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.hidden = false;
    }

    function clearError() {
        errorMessage.textContent = "";
        errorMessage.hidden = true;
    }

    function chooseFile(file) {
        clearError();

        if (!file) return;

        if (!/\.tiff?$/i.test(file.name)) {
            showError("Please select a GeoTIFF file (.tif or .tiff).");
            return;
        }

        if (file.size > 25 * 1024 * 1024) {
            showError("File exceeds the 25 MB upload limit.");
            return;
        }

        currentFile = file;
        fileName.textContent = file.name;
        selectedFile.hidden = false;
        enhanceBtn.disabled = false;
        resultsSection.hidden = true;
    }

    browseBtn.addEventListener("click", () => {
        fileInput.click();
    });

    fileInput.addEventListener("change", () => {
        chooseFile(fileInput.files[0]);
    });

    dropZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropZone.classList.remove("dragover");

        if (event.dataTransfer.files.length > 0) {
            chooseFile(event.dataTransfer.files[0]);
        }
    });

    // Creates a separate blurred/noisy PNG preview in memory.
    // The original LR file is never changed.
    function createDegradedPreview(imageUrl) {
        return new Promise((resolve, reject) => {
            const image = new Image();

            image.onload = () => {
                try {
                    const canvas = document.createElement("canvas");
                    canvas.width = image.naturalWidth;
                    canvas.height = image.naturalHeight;

                    const ctx = canvas.getContext("2d", {
                        willReadFrequently: true
                    });

                    ctx.filter = `blur(${PREVIEW_BLUR_PX}px)`;
                    ctx.drawImage(image, 0, 0);

                    const imageData = ctx.getImageData(
                        0, 0, canvas.width, canvas.height
                    );

                    const pixels = imageData.data;

                    for (let i = 0; i < pixels.length; i += 4) {
                        const noise =
                            (Math.random() * 2 - 1) *
                            PREVIEW_NOISE_STRENGTH;

                        pixels[i] = Math.max(
                            0, Math.min(255, pixels[i] + noise)
                        );
                        pixels[i + 1] = Math.max(
                            0, Math.min(255, pixels[i + 1] + noise)
                        );
                        pixels[i + 2] = Math.max(
                            0, Math.min(255, pixels[i + 2] + noise)
                        );
                    }

                    ctx.putImageData(imageData, 0, 0);
                    resolve(canvas.toDataURL("image/png"));
                } catch (error) {
                    reject(error);
                }
            };

            image.onerror = () => {
                reject(new Error("Could not load LR preview."));
            };

            image.src = imageUrl;
        });
    }

    // Keep the disclosure visible without adding a toggle.
    function addPreviewDisclosure() {
        const card = lrImage.closest(".image-card");
        if (!card || card.querySelector(".preview-disclosure")) {
            return;
        }

        const note = document.createElement("p");
        note.className = "preview-disclosure";
        note.textContent =
            "Display-only blur/noise simulation; model uses the original LR image.";

        note.style.fontSize = "11px";
        note.style.color = "#a3bbcb";
        note.style.lineHeight = "1.6";
        note.style.marginTop = "10px";

        card.appendChild(note);
    }

    enhanceBtn.addEventListener("click", async () => {
        if (!currentFile) {
            showError("Please select a satellite image first.");
            return;
        }

        clearError();
        resultsSection.hidden = true;
        processingState.hidden = false;
        enhanceBtn.disabled = true;

        const submittedFile = currentFile;
        const formData = new FormData();

        // Always upload the ORIGINAL file.
        formData.append("file", submittedFile);

        try {
            const response = await fetch("/api/enhance", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(
                    data.error || "Image enhancement failed."
                );
            }

            const lrUrl = data.lr_preview;
            const srUrl = data.sr_preview;
            const downloadUrl = data.download_url;

            if (!lrUrl || !srUrl || !downloadUrl) {
                throw new Error(
                    "Backend response is missing preview or download URLs."
                );
            }

            const cacheBust = `?v=${Date.now()}`;

            // Only the displayed LR preview is altered.
            // SR remains the actual model-generated output.
            const originalPreviewUrl = lrUrl + cacheBust;

            try {
                lrImage.src = await createDegradedPreview(
                    originalPreviewUrl
                );
            } catch (previewError) {
                console.warn(
                    "Preview effect unavailable:",
                    previewError
                );
                lrImage.src = originalPreviewUrl;
            }

            srImage.src = srUrl + cacheBust;
            downloadBtn.href = downloadUrl;

            resultFilename.textContent =
                submittedFile.name + " — 4× SR generated";

            addPreviewDisclosure();

            resultsSection.hidden = false;

            resultsSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        } catch (error) {
            console.error("GeoEnhance error:", error);
            showError(error.message);
        } finally {
            processingState.hidden = true;
            enhanceBtn.disabled = false;
        }
    });
});