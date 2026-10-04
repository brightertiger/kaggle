// Minimal Portfolio JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const filters = document.querySelector('.project-filters');
    const cards = Array.from(document.querySelectorAll('.project-card'));
    const status = document.querySelector('.filter-status');
    if (filters && status) {
        filters.hidden = false;
        const buttons = filters.querySelectorAll('button[data-filter]');
        buttons.forEach(button => {
            button.addEventListener('click', function() {
                const tier = this.dataset.filter;
                buttons.forEach(item => item.setAttribute('aria-pressed', String(item === this)));
                cards.forEach(card => {
                    const isOther = !['gold', 'silver', 'bronze'].some(medal => card.classList.contains(medal));
                    card.hidden = !(tier === 'all' || (tier === 'other' ? isOther : card.classList.contains(tier)));
                });
                status.textContent = `${cards.filter(card => !card.hidden).length} competitions shown`;
            });
        });
    }
    
    // Smooth scrolling for anchor links
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Simple fade-in animation for sections
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver(function(entries) {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, observerOptions);

        // Observe sections for fade-in effect
        const sections = document.querySelectorAll('section');
        sections.forEach(section => {
            section.style.opacity = '0';
            section.style.transform = 'translateY(20px)';
            section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            observer.observe(section);
        });
    }
    
    // Add loading state
    window.addEventListener('load', function() {
        document.body.classList.add('loaded');
    });
    
    // Console message
    console.log('🚀 Ujjwal Singh Rao - Kaggle Portfolio');
    console.log('Built with minimal design principles');
    console.log('GitHub: https://github.com/brightertiger');
    console.log('LinkedIn: https://linkedin.com/in/brightertiger');
});
