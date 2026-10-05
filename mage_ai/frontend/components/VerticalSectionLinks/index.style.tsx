import styled from 'styled-components';

import { UNIT } from '@oracle/styles/units/spacing';
import { transition } from '@oracle/styles/mixins';

export const CHART_HEIGHT_DEFAULT = UNIT * 40;

const SIDEBAR_TEXT = '#3F3F46';
const SIDEBAR_ACTIVE_BG = '#F3F4F6';

export const SectionTitleStyle = styled.div`
  color: #71717A;
  font-size: 12px;
  font-weight: 500;
  line-height: 16px;
  padding: ${UNIT}px ${UNIT}px ${UNIT / 2}px;
`;

export const ItemStyle = styled.div<{
  selected: boolean;
}>`
  ${transition()}

  align-items: center;
  border-radius: 6px;
  color: ${SIDEBAR_TEXT};
  display: flex;
  font-size: 14px;
  font-weight: 400;
  height: 40px;
  line-height: 20px;
  margin: 0 ${UNIT}px;
  padding: 0 ${UNIT}px;

  &:hover {
    background-color: ${SIDEBAR_ACTIVE_BG};
    color: ${SIDEBAR_TEXT};
  }

  ${props => props.selected && `
    background-color: ${SIDEBAR_ACTIVE_BG};
    font-weight: 500;
  `}
`;
