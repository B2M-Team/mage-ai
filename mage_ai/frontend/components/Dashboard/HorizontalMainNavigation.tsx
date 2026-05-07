import NextLink from 'next/link';
import { useMemo } from 'react';
import { useRouter } from 'next/router';

import ClientOnly from '@hocs/ClientOnly';
import {
  MainNavTabBarStyle,
  MainNavTabLinkStyle,
  MainNavTabListStyle,
} from './index.style';
import {
  DEFAULT_NAV_ITEMS,
  flattenNavigationForTabs,
  NavigationItem,
} from './VerticalNavigation';
import useProject from '@utils/models/project/useProject';

function HorizontalMainNavigation({
  navigationItems,
}: {
  navigationItems?: NavigationItem[];
}) {
  const router = useRouter();
  const { pathname } = router;

  const {
    featureEnabled,
    project,
    projectPlatformActivated,
  } = useProject();
  const defaultNavItems = useMemo(() => DEFAULT_NAV_ITEMS({
    featureEnabled,
    project,
    projectPlatformActivated,
  }), [
    featureEnabled,
    project,
    projectPlatformActivated,
  ]);

  const flatTabs = useMemo(
    () => flattenNavigationForTabs(navigationItems || defaultNavItems),
    [
      defaultNavItems,
      navigationItems,
    ],
  );

  return (
    <ClientOnly>
      <MainNavTabBarStyle aria-label="Primary">
        <MainNavTabListStyle>
          {flatTabs.map((item) => {
            const {
              disabled,
              id,
              isSelected,
              label,
              linkProps,
              onClick,
            } = item;
            const selected: boolean = isSelected
              ? isSelected(pathname, item)
              : !!pathname.match(new RegExp(`^/${id}[/]*`));

            if (!linkProps?.href) {
              return null;
            }

            return (
              <span key={id} style={{ display: 'contents' }}>
                <NextLink
                  {...linkProps}
                  passHref
                >
                  <MainNavTabLinkStyle
                    $active={selected}
                    $disabled={disabled}
                    onClick={onClick}
                  >
                    {label?.()}
                  </MainNavTabLinkStyle>
                </NextLink>
              </span>
            );
          })}
        </MainNavTabListStyle>
      </MainNavTabBarStyle>
    </ClientOnly>
  );
}

export default HorizontalMainNavigation;
