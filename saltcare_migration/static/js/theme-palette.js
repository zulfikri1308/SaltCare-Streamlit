
document.addEventListener("DOMContentLoaded", function () {
    const root = document.documentElement;
    const toggle = document.getElementById("paletteToggle");
    const panel = document.getElementById("palettePanel");
    const options = document.querySelectorAll(".palette-option");

    if (!toggle || !panel || options.length === 0) {
        console.error(
            "SaltCare: tombol, panel, atau pilihan warna tidak ditemukan."
        );
        return;
    }

    const validPalettes = [
        "ocean", "mint", "lavender", "rose", "sunset", "teal"
    ];

    function selectPalette(name) {
        const palette = validPalettes.includes(name) ? name : "ocean";

        root.setAttribute("data-palette", palette);

        options.forEach(function (option) {
            option.setAttribute(
                "aria-pressed",
                String(option.dataset.palette === palette)
            );
        });

        try {
            localStorage.setItem("saltcare-palette", palette);
        } catch (error) {
            console.warn("Preferensi warna tidak dapat disimpan.");
        }
    }

    let savedPalette = "ocean";

    try {
        savedPalette = localStorage.getItem("saltcare-palette") || "ocean";
    } catch (error) {
        // Gunakan warna default.
    }

    selectPalette(savedPalette);

    toggle.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();

        panel.hidden = !panel.hidden;

        toggle.setAttribute(
            "aria-expanded",
            String(!panel.hidden)
        );
    });

    options.forEach(function (option) {
        option.addEventListener("click", function () {
            selectPalette(option.dataset.palette);
            panel.hidden = true;
            toggle.setAttribute("aria-expanded", "false");
        });
    });

    document.addEventListener("click", function (event) {
        if (
            !panel.contains(event.target) &&
            !toggle.contains(event.target)
        ) {
            panel.hidden = true;
            toggle.setAttribute("aria-expanded", "false");
        }
    });

    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape") {
            panel.hidden = true;
            toggle.setAttribute("aria-expanded", "false");
        }
    });
});
