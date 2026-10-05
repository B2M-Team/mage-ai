import FlexContainer from '@oracle/components/FlexContainer';
import Text from '@oracle/elements/Text';
import { BlocksStacked } from '@oracle/icons';
import {
  ICON_SIZE,
  IconStyle,
  NavLinkStyle,
} from '../index.style';
import {
  NavLinkType,
} from '../constants';

type BlockNavigationProps = {
  navLinks: NavLinkType[];
  selectedLink?: NavLinkType;
  setSelectedLink: (navLink: NavLinkType) => void;
};

function BlockNavigation({
  navLinks,
  selectedLink,
  setSelectedLink,
}: BlockNavigationProps) {
  return (
    <>
      {navLinks.map((navLink: NavLinkType) => {
        const {
          Icon,
          description,
          label,
          uuid,
        } = navLink;
        const isSelected = selectedLink?.uuid === uuid;
        const IconProps = {
          fill: '#18181B',
          size: ICON_SIZE,
        };

        return (
          <NavLinkStyle
            key={uuid}
            onClick={() => setSelectedLink(navLink)}
            selected={isSelected}
          >
            <FlexContainer alignItems="center">
              {Icon && (
                <IconStyle>
                  <Icon {...IconProps} />
                </IconStyle>
              )}

              <FlexContainer alignItems="flex-start" flexDirection="column" justifyContent="center">
                <Text color="#3F3F46">
                  {label ? label() : uuid}
                </Text>

                {description && (
                  <Text muted small>
                    {description?.()}
                  </Text>
                )}
              </FlexContainer>
            </FlexContainer>
          </NavLinkStyle>
        );
      })}
    </>
  );
}

export default BlockNavigation;
