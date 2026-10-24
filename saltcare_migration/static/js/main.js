document.addEventListener("DOMContentLoaded", () => {
    const revealElements = document.querySelectorAll(".reveal");

    if ("IntersectionObserver" in window) {
        const observer = new IntersectionObserver(
            (entries, currentObserver) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add("is-visible");
                        currentObserver.unobserve(entry.target);
                    }
                });
            },
            {
                threshold: 0.12
            }
        );

        revealElements.forEach((element) => observer.observe(element));
    } else {
        revealElements.forEach((element) => {
            element.classList.add("is-visible");
        });
    }

    const navbar = document.querySelector(".navbar-salt");

    const updateNavbar = () => {
        if (navbar) {
            navbar.style.boxShadow =
                window.scrollY > 12
                    ? "0 8px 28px rgba(25, 57, 91, 0.08)"
                    : "none";
        }
    };

    updateNavbar();
    window.addEventListener("scroll", updateNavbar, { passive: true });
});


/* =========================================
   SALTCARE — THEME SWITCHER
   ========================================= */

(() => {
    const root = document.documentElement;
    const toggle = document.getElementById("themeToggle");
    const icon = document.getElementById("themeIcon");
    const label = document.getElementById("themeLabel");

    const storageKey = "saltcare-theme";

    function readSavedTheme() {
        try {
            const saved = localStorage.getItem(storageKey);

            if (saved === "light" || saved === "dark") {
                return saved;
            }
        } catch (error) {
            // Tetap berfungsi jika penyimpanan browser tidak tersedia.
        }

        return "light";
    }

    function applyTheme(theme) {
        const selectedTheme = theme === "dark" ? "dark" : "light";

        root.setAttribute("data-theme", selectedTheme);

        if (toggle && icon && label) {
            const isDark = selectedTheme === "dark";

            icon.className = isDark
                ? "bi bi-sun-fill"
                : "bi bi-moon-stars-fill";

            label.textContent = isDark ? "Light Mode" : "Dark Mode";

            toggle.setAttribute(
                "aria-label",
                isDark ? "Aktifkan light mode" : "Aktifkan dark mode"
            );

            toggle.setAttribute(
                "aria-pressed",
                String(isDark)
            );
        }

        // Sinkronkan warna bawaan browser dan komponen Bootstrap.
        document
            .querySelector('meta[name="theme-color"]')
            ?.setAttribute(
                "content",
                selectedTheme === "dark" ? "#101b20" : "#f5f9f8"
            );
    }

    applyTheme(readSavedTheme());

    toggle?.addEventListener("click", () => {
        const currentTheme = root.getAttribute("data-theme");
        const nextTheme = currentTheme === "dark" ? "light" : "dark";

        applyTheme(nextTheme);

        try {
            localStorage.setItem(storageKey, nextTheme);
        } catch (error) {
            // Tema tetap berganti untuk halaman yang sedang terbuka.
        }
    });
})();
