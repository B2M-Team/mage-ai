import styled from 'styled-components';

/**
 * Wraps the inline sidebar + graph layout below the dashboard tab bar. Clips the sidebar
 * divider, which TripleLayout draws taller than the layout, so nothing overflows the page.
 */
export const ContainerStyle = styled.div`
  overflow: hidden;
`;
