import React from 'react';

import ClickOutside from '@oracle/components/ClickOutside';
import ErrorPopup from '@components/ErrorPopup';
import ErrorsType from '@interfaces/ErrorsType';
import Flex from '@oracle/components/Flex';
import Head from '@oracle/elements/Head';
import { BreadcrumbType, MenuItemType } from '@components/shared/Header';
import HorizontalMainNavigation from './HorizontalMainNavigation';
import Subheader from './Subheader';
import TripleLayout from '@components/TripleLayout';
import { NavigationItem, VerticalNavigationProps } from './VerticalNavigation';
import {
  ContainerStyle,
} from './index.style';
import { HEADER_HEIGHT } from '@components/shared/Header/index.style';
import { ASIDE_HEADER_HEIGHT } from '@components/TripleLayout/index.style';
import useTripleLayout, {
  DEFAULT_BEFORE_RESIZE_OFFSET,
} from '@components/TripleLayout/useTripleLayout';

export type DashboardSharedProps = {
  after?: any;
  afterHeader?: any;
  afterHidden?: boolean;
  afterWidth?: number;
  afterWidthOverride?: boolean;
  before?: any;
  beforeNavigationItems?: NavigationItem[];
  beforeWidth?: number;
  beforeWidthOverride?: boolean;
  setAfterHidden?: (value: boolean) => void;
  subheaderNoPadding?: boolean;
  uuid: string;
};

type DashboardProps = {
  addProjectBreadcrumbToCustomBreadcrumbs?: boolean;
  appendBreadcrumbs?: boolean;
  beforeHeader?: any;
  breadcrumbs?: BreadcrumbType[];
  children?: any;
  contained?: boolean;
  errors?: ErrorsType;
  headerMenuItems?: MenuItemType[];
  headerOffset?: number;
  hideAfterCompletely?: boolean;
  mainContainerHeader?: any;
  setAfterWidth?: (value: number) => void;
  setBeforeWidth?: (value: number) => void;
  setErrors?: (errors: ErrorsType) => void;
  subheaderChildren?: any;
  title: string;
} & DashboardSharedProps;

function Dashboard({
  addProjectBreadcrumbToCustomBreadcrumbs,
  after,
  afterHeader,
  afterHidden,
  afterWidth,
  afterWidthOverride,
  appendBreadcrumbs,
  before,
  beforeHeader,
  beforeNavigationItems,
  beforeWidth,
  beforeWidthOverride,
  breadcrumbs: breadcrumbsProp,
  children,
  contained,
  errors,
  headerMenuItems,
  headerOffset,
  hideAfterCompletely,
  mainContainerHeader,
  navigationItems,
  setAfterHidden,
  setAfterWidth,
  setBeforeWidth,
  setErrors,
  subheaderChildren,
  subheaderNoPadding,
  title,
  uuid,
}: DashboardProps & VerticalNavigationProps, ref) {
  const {
    mainContainerRef,
    mousedownActiveAfter,
    mousedownActiveBefore,
    setMousedownActiveAfter,
    setMousedownActiveBefore,
    setWidthAfter,
    setWidthBefore,
    widthAfter,
    widthBefore,
  } = useTripleLayout(uuid, {
    beforeResizeOffset: DEFAULT_BEFORE_RESIZE_OFFSET,
    setWidthAfter: setAfterWidth,
    setWidthBefore: setBeforeWidth,
    widthAfter: afterWidth,
    widthBefore: beforeWidth,
    widthOverrideAfter: afterWidthOverride,
    widthOverrideBefore: beforeWidthOverride,
  });

  const showMainNavTabs = navigationItems?.length !== 0;
  const hasBeforeColumn = !!before || !!beforeNavigationItems?.length;
  // Header/breadcrumb strip is gone. Cancel TripleLayout's built-in header offset
  // so sidebars and the main column start together, and tabs sit in the content.
  const layoutTopOffset = HEADER_HEIGHT;
  const tripleLayoutHeaderOffset = (headerOffset ?? 0) - ASIDE_HEADER_HEIGHT;

  return (
    <>
      <Head title={title} />

      <ContainerStyle ref={ref}>
        <Flex
          flex={1}
          flexDirection="column"
        >
          {/* @ts-ignore */}
          <TripleLayout
            after={after}
            afterHeader={afterHeader}
            afterHeightOffset={layoutTopOffset}
            afterHidden={afterHidden}
            afterMousedownActive={mousedownActiveAfter}
            afterWidth={widthAfter}
            before={before || (beforeNavigationItems?.length ? <></> : null)}
            beforeHeader={beforeHeader}
            beforeHeightOffset={layoutTopOffset}
            beforeMousedownActive={mousedownActiveBefore}
            beforeNavigationItems={beforeNavigationItems}
            beforeWidth={hasBeforeColumn ? widthBefore : 0}
            navigationShowMore={!!beforeNavigationItems?.length}
            contained={contained}
            headerOffset={tripleLayoutHeaderOffset}
            hideAfterCompletely={!after || hideAfterCompletely}
            leftOffset={0}
            mainContainerHeader={mainContainerHeader}
            mainContainerRef={mainContainerRef}
            setAfterHidden={setAfterHidden}
            setAfterMousedownActive={setMousedownActiveAfter}
            setAfterWidth={setWidthAfter}
            setBeforeMousedownActive={setMousedownActiveBefore}
            setBeforeWidth={beforeWidthOverride ? undefined : setWidthBefore}
          >
            {showMainNavTabs && (
              <HorizontalMainNavigation
                navigationItems={navigationItems}
              />
            )}

            {subheaderChildren && (
              <Subheader noPadding={subheaderNoPadding}>
                {subheaderChildren}
              </Subheader>
            )}

            {children}
          </TripleLayout>
        </Flex>
      </ContainerStyle>

      {errors && (
        <ClickOutside
          disableClickOutside
          isOpen
          onClickOutside={() => setErrors?.(null)}
        >
          <ErrorPopup
            {...errors}
            onClose={() => setErrors?.(null)}
          />
        </ClickOutside>
      )}
    </>
  );
}

export default React.forwardRef(Dashboard);
