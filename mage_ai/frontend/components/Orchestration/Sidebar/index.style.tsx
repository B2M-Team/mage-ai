import styled from 'styled-components';

import light from '@oracle/styles/themes/light';
import { UNIT } from '@oracle/styles/units/spacing';

export const EntryStyle = styled.div<any>`
  border-bottom: 1px solid ${props => (props.theme.borders || light.borders).medium};

  background: ${props => (props.theme.background || light.background).panel};
  padding: ${2 * UNIT}px;

  ${props => props.selected && `
    background: ${(props.theme.background || light.background).muted};
  `}
`;
