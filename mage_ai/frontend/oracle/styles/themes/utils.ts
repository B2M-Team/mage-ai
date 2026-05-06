// @ts-ignore
import Cookies from 'js-cookie';

import light from '@oracle/styles/themes/light';
import { SHARED_OPTS } from '@api/utils/token';

export const LOCAL_STORAGE_KEY_THEME: 'current_theme' = 'current_theme';
const LOCAL_STORAGE_KEY_THEME_LIGHT: number = 1;

/** Always light theme for this deployment. */
export function getCurrentTheme(_ctx?: any, _invertedTheme?: number) {
  return light;
}

export function getCurrentInvertedTheme(_ctx?: any) {
  return light;
}

export function setCurrentTheme(_theme?: unknown) {
  // @ts-ignore
  Cookies.set(LOCAL_STORAGE_KEY_THEME, LOCAL_STORAGE_KEY_THEME_LIGHT, { ...SHARED_OPTS, expires: 9999 });
}

export function toggleTheme() {
  setCurrentTheme();
}
