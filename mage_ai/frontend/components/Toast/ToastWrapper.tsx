import styled from 'styled-components';
import { ToastContainer, toast } from 'react-toastify';

/**
 * Toasts match the B2Metric product (react-toastify v9 default light theme): white card,
 * grey text, an icon and progress bar in the type's color, shown at the top center.
 */
const TOAST_TYPES = {
  error: {
    color: '#e74c3c',
    iconPath: [
      'M11.983 0a12.206 12.206 0 00-8.51 3.653A11.8 11.8 0 000 12.207 11.779 11.779 0 0011.8 24h.214',
      'A12.111 12.111 0 0024 11.791 11.766 11.766 0 0011.983 0zM10.5 16.542a1.476 1.476 0 011.449-1.53',
      'h.027a1.527 1.527 0 011.523 1.47 1.475 1.475 0 01-1.449 1.53h-.027a1.529 1.529 0 01-1.523-1.47z',
      'M11 12.5v-6a1 1 0 012 0v6a1 1 0 11-2 0z',
    ].join(''),
  },
  info: {
    color: '#3498db',
    iconPath: [
      'M12 0a12 12 0 1012 12A12.013 12.013 0 0012 0zm.25 5a1.5 1.5 0 11-1.5 1.5 1.5 1.5 0 011.5-1.5z',
      'm2.25 13.5h-4a1 1 0 010-2h.75a.25.25 0 00.25-.25v-4.5a.25.25 0 00-.25-.25h-.75a1 1 0 010-2h1',
      'a2 2 0 012 2v4.75a.25.25 0 00.25.25h.75a1 1 0 110 2z',
    ].join(''),
  },
  success: {
    color: '#07bc0c',
    iconPath: [
      'M12 0a12 12 0 1012 12A12.014 12.014 0 0012 0zm6.927 8.2l-6.845 9.289a1.011 1.011 0 01-1.43.188',
      'l-4.888-3.908a1 1 0 111.25-1.562l4.076 3.261 6.227-8.451a1 1 0 111.61 1.183z',
    ].join(''),
  },
  warning: {
    color: '#f1c40f',
    iconPath: [
      'M23.32 17.191L15.438 2.184C14.728.833 13.416 0 11.996 0c-1.42 0-2.733.833-3.443 2.184',
      'L.533 17.448a4.744 4.744 0 000 4.368C1.243 23.167 2.555 24 3.975 24h16.05',
      'C22.22 24 24 22.044 24 19.632c0-.904-.251-1.746-.68-2.44zm-9.622 1.46',
      'c0 1.033-.724 1.823-1.698 1.823s-1.698-.79-1.698-1.822v-.043c0-1.028.724-1.822 1.698-1.822',
      's1.698.79 1.698 1.822v.043zm.039-12.285l-.84 8.06',
      'c-.057.581-.408.943-.897.943-.49 0-.84-.367-.896-.942l-.84-8.065c-.057-.624.25-1.095.779-1.095',
      'h1.91c.528.005.84.476.784 1.1z',
    ].join(''),
  },
};

function iconUrl(color: string, path: string): string {
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path fill='${color}' d='${path}'/></svg>`;

  return `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
}

const ToastWrapper = styled.div`
  .Toastify__toast-container {
    /* The product's toasts inherit 16px / 1.5 from the page; em offsets depend on it too. */
    font-size: 16px;
    line-height: 1.5;
    min-width: fit-content;
  }

  .Toastify__toast {
    background: #fff;
    border-radius: 4px;
    color: #757575;
  }

  .Toastify__toast-body {
    align-items: center;
    display: flex;
    padding: 6px;
  }

  .Toastify__close-button {
    color: #000;
    opacity: 0.3;

    &:focus,
    &:hover {
      opacity: 1;
    }
  }

  ${Object.entries(TOAST_TYPES).map(([type, { color, iconPath }]) => `
    .Toastify__toast--${type} .Toastify__toast-body::before {
      background: ${iconUrl(color, iconPath)} center / contain no-repeat;
      content: '';
      flex-shrink: 0;
      height: 20px;
      margin-inline-end: 10px;
      width: 20px;
    }

    .Toastify__progress-bar--${type} {
      background: ${color};
    }
  `).join('')}
`;

export default function Toast() {
  return (
    <ToastWrapper>
      <ToastContainer
        position={toast.POSITION.TOP_CENTER}
      />
    </ToastWrapper>
  );
}
