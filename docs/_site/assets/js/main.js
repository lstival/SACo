// Remote Sensing Animated Background
const canvas = document.getElementById('remote-sensing-bg');
const ctx = canvas.getContext('2d');

// Set canvas size
function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Satellite orbit paths
class SatelliteOrbit {
    constructor() {
        this.angle = Math.random() * Math.PI * 2;
        this.radius = Math.random() * 200 + 100;
        this.speed = (Math.random() * 0.0005 + 0.0002) * (Math.random() > 0.5 ? 1 : -1);
        this.centerX = Math.random() * canvas.width;
        this.centerY = Math.random() * canvas.height;
        this.opacity = Math.random() * 0.3 + 0.1;
    }

    update() {
        this.angle += this.speed;
    }

    draw() {
        const x = this.centerX + Math.cos(this.angle) * this.radius;
        const y = this.centerY + Math.sin(this.angle) * this.radius;

        // Draw orbit path
        ctx.beginPath();
        ctx.arc(this.centerX, this.centerY, this.radius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(59, 130, 246, ${this.opacity * 0.2})`;
        ctx.lineWidth = 1;
        ctx.stroke();

        // Draw satellite point
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(139, 92, 246, ${this.opacity})`;
        ctx.fill();
    }
}

// Grid pattern (representing multispectral bands)
class GridPattern {
    constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 100 + 50;
        this.rotation = Math.random() * Math.PI;
        this.rotationSpeed = (Math.random() * 0.0002 + 0.0001) * (Math.random() > 0.5 ? 1 : -1);
        this.opacity = Math.random() * 0.15 + 0.05;
        this.color = Math.random() > 0.5 ? '59, 130, 246' : '139, 92, 246'; // Blue or purple
    }

    update() {
        this.rotation += this.rotationSpeed;
    }

    draw() {
        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(this.rotation);

        // Draw grid cells
        const cellSize = this.size / 4;
        for (let i = -2; i < 2; i++) {
            for (let j = -2; j < 2; j++) {
                ctx.strokeStyle = `rgba(${this.color}, ${this.opacity})`;
                ctx.lineWidth = 0.5;
                ctx.strokeRect(i * cellSize, j * cellSize, cellSize, cellSize);
            }
        }

        ctx.restore();
    }
}

// Scanning lines (like satellite scanning)
class ScanLine {
    constructor() {
        this.y = Math.random() * canvas.height;
        this.speed = Math.random() * 0.3 + 0.1;
        this.opacity = Math.random() * 0.2 + 0.1;
        this.width = canvas.width;
    }

    update() {
        this.y += this.speed;
        if (this.y > canvas.height) {
            this.y = -10;
        }
    }

    draw() {
        const gradient = ctx.createLinearGradient(0, this.y - 5, 0, this.y + 5);
        gradient.addColorStop(0, 'rgba(59, 130, 246, 0)');
        gradient.addColorStop(0.5, `rgba(59, 130, 246, ${this.opacity})`);
        gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, this.y - 5, this.width, 10);
    }
}

// Initialize elements
const satellites = Array.from({ length: 3 }, () => new SatelliteOrbit());
const grids = Array.from({ length: 5 }, () => new GridPattern());
const scanLines = Array.from({ length: 2 }, () => new ScanLine());

// Animation loop
function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Update and draw all elements
    grids.forEach(grid => {
        grid.update();
        grid.draw();
    });

    satellites.forEach(satellite => {
        satellite.update();
        satellite.draw();
    });

    scanLines.forEach(line => {
        line.update();
        line.draw();
    });

    requestAnimationFrame(animate);
}

animate();

// Dark mode toggle
const themeToggle = document.getElementById('theme-toggle');
const html = document.documentElement;

// Check for saved theme preference or default to light mode
const currentTheme = localStorage.getItem('theme') || 'light';
html.setAttribute('data-theme', currentTheme);

themeToggle.addEventListener('click', () => {
    const theme = html.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    html.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
});

// Mobile menu toggle
const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
const navLinks = document.querySelector('.nav-links');

if (mobileMenuToggle) {
    mobileMenuToggle.addEventListener('click', () => {
        navLinks.classList.toggle('active');
    });
}

// Copy to clipboard functionality
document.querySelectorAll('.copy-btn').forEach(button => {
    button.addEventListener('click', async () => {
        const targetId = button.getAttribute('data-target');
        const targetElement = document.getElementById(targetId);

        if (targetElement) {
            try {
                await navigator.clipboard.writeText(targetElement.textContent);

                // Visual feedback
                const originalText = button.innerHTML;
                button.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    Copied!
                `;
                button.classList.add('copied');

                setTimeout(() => {
                    button.innerHTML = originalText;
                    button.classList.remove('copied');
                }, 2000);
            } catch (err) {
                console.error('Failed to copy:', err);
            }
        }
    });
});

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add fade-in animation on scroll
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe sections for fade-in effect
document.querySelectorAll('.section').forEach(section => {
    section.style.opacity = '0';
    section.style.transform = 'translateY(20px)';
    section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(section);
});
