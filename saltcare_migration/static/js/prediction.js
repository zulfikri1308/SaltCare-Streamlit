document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictionForm");
    const notice = document.getElementById("formNotice");

    if (!form || !notice) return;

    const fields = Array.from(
        form.querySelectorAll(".feature-field")
    );

    function showNotice(message, type) {
        notice.textContent = message;
        notice.className = `form-notice visible ${type}`;
    }

    function clearNotice() {
        notice.textContent = "";
        notice.className = "form-notice";
    }

    function validateField(field) {
        const control = field.querySelector("input, select");
        const error = field.querySelector(".field-error");

        field.classList.remove("invalid");
        error.textContent = "";

        if (!control.value.trim()) {
            field.classList.add("invalid");
            error.textContent = "Kolom ini wajib diisi.";
            return false;
        }

        if (control instanceof HTMLInputElement) {
            const value = Number(control.value);
            const min = control.min === "" ? -Infinity : Number(control.min);
            const max = control.max === "" ? Infinity : Number(control.max);

            if (!Number.isFinite(value)) {
                field.classList.add("invalid");
                error.textContent = "Masukkan angka yang valid.";
                return false;
            }

            if (value < min || value > max) {
                field.classList.add("invalid");
                error.textContent =
                    `Nilai harus berada di antara ${min} dan ${max}.`;
                return false;
            }

            if (control.validity.stepMismatch) {
                field.classList.add("invalid");
                error.textContent = "Periksa kembali format angka.";
                return false;
            }
        }

        return true;
    }

    fields.forEach((field) => {
        const control = field.querySelector("input, select");

        control.addEventListener("input", () => {
            if (field.classList.contains("invalid")) {
                validateField(field);
            }
            clearNotice();
        });

        control.addEventListener("change", () => {
            if (field.classList.contains("invalid")) {
                validateField(field);
            }
            clearNotice();
        });
    });

    form.addEventListener("submit", (event) => {
        event.preventDefault();

        let firstInvalid = null;
        let allValid = true;

        fields.forEach((field) => {
            if (!validateField(field)) {
                allValid = false;
                if (!firstInvalid) firstInvalid = field;
            }
        });

        if (!allValid) {
            showNotice(
                "Masih ada kolom yang perlu diperiksa. Lengkapi atau perbaiki kolom yang ditandai.",
                "error"
            );

            firstInvalid?.querySelector("input, select")?.focus();
            firstInvalid?.scrollIntoView({
                behavior: "smooth",
                block: "center"
            });
            return;
        }

        showNotice(
            "Semua kolom sudah terisi dengan format yang sesuai. Formulir siap untuk tahap integrasi model; prediksi belum dijalankan.",
            "success"
        );
    });

    form.addEventListener("reset", () => {
        window.setTimeout(() => {
            fields.forEach((field) => {
                field.classList.remove("invalid");
                field.querySelector(".field-error").textContent = "";
            });

            clearNotice();
        }, 0);
    });
});