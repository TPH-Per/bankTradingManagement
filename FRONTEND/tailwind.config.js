/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./main.js",
    "./src/**/*.{js,html}",
  ],
  theme: {
    extend: {
      colors: {
        'primary': {
          DEFAULT: 'hsl(160, 80%, 45%)', // A vibrant teal/green
          'light': 'hsl(160, 80%, 95%)', // Very light for dark theme hover
          'dark': 'hsl(160, 80%, 35%)',
        },
        'secondary': {
          DEFAULT: 'hsl(220, 10%, 65%)', // Lighter gray for text
          'light': 'hsl(220, 15%, 20%)',
          'dark': 'hsl(220, 10%, 85%)',
        },
        'background': 'hsl(220, 20%, 12%)', // Dark slate background
        'surface': 'hsl(220, 20%, 16%)', // Slightly lighter for cards
        'border': 'hsl(220, 15%, 25%)',
        'success': 'hsl(145, 70%, 50%)',
        'warning': 'hsl(45, 90%, 60%)',
        'danger': 'hsl(0, 80%, 60%)',
        'info': 'hsl(190, 80%, 60%)',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      boxShadow: {
        'subtle': '0 4px 12px rgba(0, 0, 0, 0.1)',
        'md': '0 0 20px rgba(0, 0, 0, 0.2)',
      },
      borderRadius: {
        'xl': '1rem',
        '2xl': '1.5rem',
      },
      keyframes: {
        'shimmer': {
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        // Removed old animations to prevent conflicts with anime.js
      },
    },
  },
  plugins: [],
}
