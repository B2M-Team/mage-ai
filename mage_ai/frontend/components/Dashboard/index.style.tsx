import React, { useRef, useState } from 'react';
import styled from 'styled-components';

import Spacing from '@oracle/elements/Spacing';
import dark from '@oracle/styles/themes/dark';
import { BORDER_RADIUS } from '@oracle/styles/units/borders';
import { HEADER_HEIGHT } from '@components/shared/Header/index.style';
import { PADDING_UNITS, UNIT } from '@oracle/styles/units/spacing';
import { ScrollbarStyledCss } from '@oracle/styles/scrollbars';
import { transition } from '@oracle/styles/mixins';

export const VERTICAL_NAVIGATION_WIDTH = (PADDING_UNITS * UNIT) + (5 * UNIT) + (PADDING_UNITS * UNIT) + 1;

/**
 * Matches where the first breadcrumb label starts:
 * Header padding-left (2*UNIT) + Breadcrumbs first Spacing ml={2} (theme.space[2] === 2*UNIT).
 */
export const MAIN_NAV_TAB_ROW_INSET_LEFT = 4 * UNIT;

/** Fixed strip under the app header for primary horizontal nav tabs */
export const MAIN_NAV_TAB_BAR_HEIGHT = Math.round(5 * UNIT);

/** No extra gap between the fixed tab bar and the scrollable content area. */
export const MAIN_NAV_TAB_BAR_MARGIN_BOTTOM = 0;

/**
 * Vertical offset for the content container — 40px tab row plus the 1px border.
 */
export const MAIN_NAV_TAB_BAR_LAYOUT_HEIGHT = MAIN_NAV_TAB_BAR_HEIGHT + 1;

export const ContainerStyle = styled.div<{
  $withMainNavTabs?: boolean;
}>`
  display: flex;
  flex-direction: row;
  height: calc(100vh - ${HEADER_HEIGHT}px - ${props =>
    props.$withMainNavTabs ? MAIN_NAV_TAB_BAR_LAYOUT_HEIGHT : 0}px);
  position: fixed;
  top: ${props =>
    HEADER_HEIGHT + (props.$withMainNavTabs ? MAIN_NAV_TAB_BAR_LAYOUT_HEIGHT : 0)}px;
  width: 100%;

  ${props => `
    background-color: ${(props.theme.background || dark.background).page};
  `}
`;

export const MainNavTabBarStyle = styled.nav`
  background-color: #FFFFFF;
  border-bottom: 1px solid ${props => (props.theme.borders || dark.borders).medium};
  box-sizing: border-box;
  display: flex;
  flex-direction: row;
  overflow-x: auto;
  padding: ${UNIT}px ${2 * UNIT}px 0;
  width: 100%;
  ${ScrollbarStyledCss}
`;

export const MainNavTabListStyle = styled.div`
  align-items: flex-end;
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  gap: ${UNIT / 2}px;
`;

export const MainNavTabLinkStyle = styled.a<{
  $active?: boolean;
  $disabled?: boolean;
}>`
  align-items: center;
  border-bottom: 2px solid transparent;
  box-sizing: border-box;
  color: #a9a9ab;
  cursor: pointer;
  display: inline-flex;
  font-size: 14px;
  font-weight: 600;
  line-height: 20px;
  margin-bottom: -1px;
  padding: ${2 * UNIT}px ${UNIT}px;
  text-decoration: none;
  white-space: nowrap;
  ${transition()}

  ${props => props.$disabled && `
    cursor: not-allowed;
    opacity: 0.45;
    pointer-events: none;
  `}

  ${props => props.$active && `
    border-bottom-color: #18181B;
    color: #18181B;
  `}
`;

type VerticalNavigationStyleProps = {
  aligned?: 'left' | 'right';
  borderless?: boolean;
  children?: any;
  showMore?: boolean;
};

const VerticalNavigationStyleComponent = styled.div<VerticalNavigationStyleProps & {
  visible?: boolean;
}>`
  height: 100%;

  ${props => `
    background-color: ${(props.theme.background || dark.background).panel};
  `}

  ${props => !props.borderless && props.aligned !== 'right' && `
    border-right: 1px solid ${(props.theme.borders || dark.borders).medium};
  `}

  ${props => !props.borderless && props.aligned === 'right' && `
    border-left: 1px solid ${(props.theme.borders || dark.borders).medium};
  `}

  @keyframes animate-in {
    0% {
      width: ${UNIT * 21}px;
    }

    100% {
      width: ${UNIT * 34}px;
    }
  }

  ${props => props.showMore && props.visible && `
    &:hover {
      animation: animate-in 100ms linear forwards;
      position: fixed;
      z-index: 100;
    }
  `}

  ${props => props.showMore && props.visible && props.aligned === 'right' && `
    &:hover {
      right: 0;
      top: ${HEADER_HEIGHT}px;
    }
  `}
`;

export function VerticalNavigationStyle({
  aligned,
  borderless,
  children,
  showMore,
}: {
  children: any;
} & VerticalNavigationStyleProps) {
  const timeout = useRef(null);
  const [visible, setVisible] = useState<boolean>(false);

  return (
    <VerticalNavigationStyleComponent
      aligned={aligned}
      borderless={borderless && !visible}
      onMouseEnter={showMore
        ? () => {
          clearTimeout(timeout.current);
          timeout.current = setTimeout(() => {
            setVisible(true);
          }, 300);
        }
        : null
      }
      onMouseLeave={showMore
        ? () => {
          clearTimeout(timeout.current);
          setVisible(false);
        }
        : null
      }
      showMore={showMore}
      visible={visible}
    >
      <Spacing
        px={showMore && visible ? 0 : PADDING_UNITS}
        py={showMore && visible ? 1 : PADDING_UNITS}
      >
        {React.cloneElement(children, {
          showMore,
          visible,
        })}
      </Spacing>
    </VerticalNavigationStyleComponent>
  );
}

export const SubheaderStyle = styled.div<{
  noPadding?: boolean;
}>`
  position: sticky;
  top: 0;
  width: 100%;
  z-index: 3;

  ${props => `
    background-color: ${(props.theme.background || dark.background).page};
    border-bottom: 1px solid ${(props.theme.borders || dark.borders).light};
  `}

  ${props => !props.noPadding && `
    padding: ${PADDING_UNITS * UNIT}px;
  `}
`;

export const ContentStyle = styled.div<{
  heightOffset?: number;
}>`
  ${ScrollbarStyledCss}

  overflow: auto;

  ${props => `
    height: calc(100vh - ${HEADER_HEIGHT + (props.heightOffset || 0)}px);
  `}
`;

export const NavigationItemStyle = styled.div<{
  primary?: boolean;
  selected?: boolean;
  showMore?: boolean;
  withGradient?: boolean;
}>`
  align-items: center;
  border-radius: ${BORDER_RADIUS}px;
  display: flex;
  height: ${UNIT * 5}px;
  justify-content: center;
  padding: ${UNIT}px;
  width: ${UNIT * 5}px;

  ${props => props.primary && `
    ${transition()}
    background: ${(props.theme || dark).chart.backgroundPrimary};
    border: 1px solid ${(props.theme || dark).feature.active};

    &:hover {
      background-color: ${(props.theme || dark).interactive.linkSecondary};
    }
  `}

  ${props => props.selected && !props.withGradient && `
    background-color: #F3F4F6;
  `}

  ${props => props.selected && props.withGradient && `
    background-color: ${(props.theme.background || dark.background).codeTextarea};
  `}

  ${props => !props.selected && props.showMore &&`
    background-color: ${(props.theme.interactive || dark.interactive).defaultBackground};
  `}
`;

export const NavigationLinkStyle = styled.a<{
  selected?: boolean;
}>`
  ${transition()}

  border-radius: 6px;
  color: #3F3F46;
  display: block;
  font-size: 14px;
  font-weight: 400;
  line-height: 20px;
  margin: 0 ${UNIT}px;
  padding: ${UNIT}px;

  p {
    font-weight: inherit;
  }

  &:hover {
    background-color: #F3F4F6;
    color: #3F3F46;
  }

  ${props => props.selected && `
    background-color: #F3F4F6;
    font-weight: 500;
  `}
`;

export const ImageStyle = styled.div<{
  imageUrl: string;
  size?: number;
}>`
  background-position: 0 0;
  background-repeat: no-repeat;
  background-size: contain;
  height: ${UNIT * 12}px;
  width: ${UNIT * 12}px;

  ${props => `
    background-image: url(${props.imageUrl});
  `}

  ${props => props.size && `
    height: ${props.size}px;
    width: ${props.size}px;
  `}
`;
