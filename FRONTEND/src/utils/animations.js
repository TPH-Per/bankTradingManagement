import anime from 'animejs/lib/anime.es.js';

/**
 * Animates a page element sliding in.
 * @param {HTMLElement} element The element to animate.
 */
export function animatePageIn(element) {
  anime({
    targets: element,
    opacity: [0, 1],
    translateY: [20, 0],
    duration: 400,
    easing: 'easeOutCubic',
  });
}

/**
 * Animates a page element sliding out.
 * @param {HTMLElement} element The element to animate.
 * @param {Function} onComplete Callback function to execute after animation.
 */
export function animatePageOut(element, onComplete) {
  anime({
    targets: element,
    opacity: [1, 0],
    translateY: [0, -20],
    duration: 300,
    easing: 'easeInCubic',
    complete: onComplete,
  });
}

/**
 * Animates a list of elements with a staggering slide-in effect.
 * @param {string | HTMLElement[]} targets The selector or elements to animate.
 */
export function animateStaggerIn(targets) {
  anime({
    targets: targets,
    opacity: [0, 1],
    translateY: [15, 0],
    delay: anime.stagger(60, { start: 100 }),
    duration: 400,
    easing: 'easeOutCubic',
  });
}

/**
 * Animates a modal opening.
 * @param {HTMLElement} overlay The modal overlay element.
 * @param {HTMLElement} modal The modal container element.
 */
export function animateModalOpen(overlay, modal) {
  anime.timeline({
    easing: 'easeOutCubic',
  }).add({
    targets: overlay,
    opacity: [0, 1],
    duration: 300,
  }).add({
    targets: modal,
    opacity: [0, 1],
    scale: [0.9, 1],
    duration: 300,
  }, '-=200');
}

/**
 * Animates a modal closing.
 * @param {HTMLElement} overlay The modal overlay element.
 * @param {HTMLElement} modal The modal container element.
 * @param {Function} onComplete Callback function to execute after animation.
 */
export function animateModalClose(overlay, modal, onComplete) {
  anime.timeline({
    easing: 'easeInCubic',
    complete: onComplete,
  }).add({
    targets: modal,
    opacity: 0,
    scale: 0.9,
    duration: 200,
  }).add({
    targets: overlay,
    opacity: 0,
    duration: 200,
  }, '-=100');
}

/**
 * Shows a temporary success or error message.
 * @param {HTMLElement} el The element to display the message in.
 * @param {string} message The message text.
 * @param {boolean} isSuccess Whether the message is for success or error.
 */
export function showResponseMessage(el, message, isSuccess) {
    if (!el) return;
    el.textContent = message;
    el.className = `text-center text-sm mt-2 ${isSuccess ? 'text-success' : 'text-danger'}`;
    anime({
        targets: el,
        opacity: [0, 1],
        translateY: [-10, 0],
        duration: 300,
        complete: () => {
            setTimeout(() => {
                anime({
                    targets: el,
                    opacity: 0,
                    duration: 500,
                    complete: () => el.textContent = ''
                });
            }, 3000);
        }
    });
}
