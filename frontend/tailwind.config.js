/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          'Roboto',
          '"Hiragino Sans"',
          '"BIZ UDPGothic"',
          '"BIZ UDGothic"',
          'Meiryo',
          'sans-serif',
        ],
        mono: [
          '"JetBrains Mono"',
          'ui-monospace',
          'SFMono-Regular',
          'Consolas',
          '"Liberation Mono"',
          '"Hiragino Sans"',
          '"BIZ UDGothic"',
          '"BIZ UDPGothic"',
          'Meiryo',
          'monospace',
        ],
      },
      colors: {
        studio: {
          950: '#09090B',
          900: '#111113',
          850: '#16161A',
          800: '#1E1E24',
          700: '#27272A',
        },
      },
      backgroundImage: {
        'aurora-horizon':
          'linear-gradient(to top, rgba(157, 23, 77, 0.24) 0%, rgba(134, 25, 143, 0.14) 20%, rgba(88, 28, 135, 0.05) 40%, rgba(9, 9, 11, 0) 58%)',
        'aurora-core':
          'radial-gradient(82% 50% at 50% 106%, rgba(225, 29, 72, 0.22) 0%, rgba(190, 24, 93, 0.17) 34%, rgba(147, 51, 234, 0.08) 64%, rgba(9, 9, 11, 0) 84%)',
        'aurora-curtains':
          'radial-gradient(58% 44% at 18% 104%, rgba(162, 28, 175, 0.19) 0%, rgba(126, 34, 206, 0.09) 50%, rgba(9, 9, 11, 0) 78%), radial-gradient(60% 45% at 82% 105%, rgba(190, 24, 93, 0.19) 0%, rgba(168, 85, 247, 0.09) 50%, rgba(9, 9, 11, 0) 78%)',
      },
    },
  },
  plugins: [],
};
