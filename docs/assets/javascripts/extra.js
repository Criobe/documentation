/* Custom JavaScript for CRIOBE Documentation */

// Add copy button feedback
document.addEventListener('DOMContentLoaded', function() {
  // Enhanced copy button feedback
  document.querySelectorAll('button[data-clipboard-text]').forEach(function(button) {
    button.addEventListener('click', function() {
      // Show temporary success message
      const originalTitle = button.getAttribute('title');
      button.setAttribute('title', 'Copied!');
      setTimeout(function() {
        button.setAttribute('title', originalTitle);
      }, 2000);
    });
  });

  // Add anchor links to headings (for older browsers)
  document.querySelectorAll('h2, h3, h4').forEach(function(heading) {
    if (heading.id) {
      const link = document.createElement('a');
      link.className = 'headerlink';
      link.href = '#' + heading.id;
      link.title = 'Permanent link';
      link.innerHTML = '&para;';
      heading.appendChild(link);
    }
  });

  // Smooth scrolling for internal links
  document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
    anchor.addEventListener('click', function(e) {
      const targetId = this.getAttribute('href').substring(1);
      const targetElement = document.getElementById(targetId);
      if (targetElement) {
        e.preventDefault();
        targetElement.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });
});

// Add external link indicators
window.addEventListener('load', function() {
  document.querySelectorAll('a[href^="http"]').forEach(function(link) {
    if (!link.hostname.includes(window.location.hostname)) {
      link.setAttribute('target', '_blank');
      link.setAttribute('rel', 'noopener noreferrer');
      // Add external link icon if desired
      // link.classList.add('external-link');
    }
  });
});

// Table of contents highlighting enhancement
let scrollTimeout;
window.addEventListener('scroll', function() {
  clearTimeout(scrollTimeout);
  scrollTimeout = setTimeout(function() {
    // Update TOC highlighting based on scroll position
    const headers = document.querySelectorAll('h2, h3, h4');
    let currentSection = null;

    headers.forEach(function(header) {
      const rect = header.getBoundingClientRect();
      if (rect.top <= 100) {
        currentSection = header.id;
      }
    });

    if (currentSection) {
      document.querySelectorAll('.md-nav__link--active').forEach(function(el) {
        el.classList.remove('md-nav__link--active');
      });
      const tocLink = document.querySelector(`.md-nav__link[href="#${currentSection}"]`);
      if (tocLink) {
        tocLink.classList.add('md-nav__link--active');
      }
    }
  }, 100);
});
