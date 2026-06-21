// Theme toggle - dark/light mode with smooth transition + chart sync
(function() {
    const html = document.documentElement;

    function currentTheme() {
        return html.getAttribute('data-theme') || 'light';
    }

    function applyTheme(theme) {
        html.setAttribute('data-theme', theme);
        try { localStorage.setItem('theme', theme); } catch (e) {}
        document.querySelectorAll('#theme-icon').forEach(function(icon) {
            icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        });
        window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: theme } }));
    }

    applyTheme(currentTheme());

    document.querySelectorAll('#theme-toggle').forEach(function(btn) {
        btn.addEventListener('click', function() {
            const next = currentTheme() === 'dark' ? 'light' : 'dark';
            applyTheme(next);
        });
    });

    if (window.matchMedia) {
        const mq = window.matchMedia('(prefers-color-scheme: dark)');
        try {
            mq.addEventListener('change', function(e) {
                if (!localStorage.getItem('theme')) {
                    applyTheme(e.matches ? 'dark' : 'light');
                }
            });
        } catch (err) {}
    }
})();
