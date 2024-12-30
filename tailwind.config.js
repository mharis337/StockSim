/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./node_modules/flowbite/**/*.js",
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#CDB4DB',
        secondary: '#A2D2FF',
        accent: '#BDE0FE',
      }
    },
  },
  plugins: [
    require('flowbite/plugin')
],
}

