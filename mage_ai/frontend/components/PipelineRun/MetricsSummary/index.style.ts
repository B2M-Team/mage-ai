import styled from 'styled-components';

import dark from '@oracle/styles/themes/dark';
import { FONT_FAMILY_REGULAR } from '@oracle/styles/fonts/primary';
import { BORDER_RADIUS, BORDER_WIDTH, BORDER_STYLE } from '@oracle/styles/units/borders';
import { UNIT } from '@oracle/styles/units/spacing';

export const MetricCountText = styled.span<{ danger?: boolean }>`
  font-family: ${FONT_FAMILY_REGULAR};
  font-size: 32px;
  font-weight: 700;
  line-height: 38px;

  ${props => `
    color: ${props.danger
      ? (props.theme.interactive || dark.interactive).dangerBorder
      : (props.theme.content || dark.content).active};
  `}
`;

export const MetricsSummaryContainerStyle = styled.div`
  width: 100%;
  padding: ${UNIT * 2}px;
  border-radius: ${BORDER_RADIUS}px;

  ${props => `
    border: ${BORDER_WIDTH}px ${BORDER_STYLE} ${(props.theme || dark).interactive.defaultBorder};
    box-shadow: ${(props.theme || dark).shadow.frame};
  `}

`;

export const MetricContainerStyle = styled.div<{
  includeLeftBorder?: boolean;
}>`
  display: flex;
  flex: 1;
  align-items: center;
  justify-content: space-between;

  ${props => props.includeLeftBorder && `
    @media (min-width: 1200px) {
      padding-left: 24px;
      border-left: ${BORDER_WIDTH}px ${BORDER_STYLE} ${(props.theme || dark).interactive.defaultBorder};
    }
  `}
`;
