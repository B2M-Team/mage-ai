import React from 'react';

import ClickOutside from '@oracle/components/ClickOutside';
import ErrorPopup from '@components/ErrorPopup';
import ErrorsType from '@interfaces/ErrorsType';
import Flex from '@oracle/components/Flex';
import Head from '@oracle/elements/Head';
import Header, { BreadcrumbType, MenuItemType } from '@components/shared/Header';
import HorizontalMainNavigation from './HorizontalMainNavigation';
import Subheader from './Subheader';
import TripleLayout from '@components/TripleLayout';
import { VerticalNavigationProps } from './VerticalNavigation';
import useProject from '@utils/models/project/useProject';
import {
  ContainerStyle,
  MAIN_NAV_TAB_BAR_LAYOUT_HEIGHT,
} from './index.style';
import { HEADER_HEIGHT } from '@components/shared/Header/index.style';
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
  beforeWidth?: number;
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
  beforeWidth,
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
  });

  const {
    project,
  } = useProject();

  const breadcrumbs = [];
  if (breadcrumbsProp) {
    // if (addProjectBreadcrumbToCustomBreadcrumbs) {
    //   breadcrumbs.push(...breadcrumbProjects);
    // }

    breadcrumbs.push(...breadcrumbsProp);
  }

  if ((!breadcrumbsProp?.length || appendBreadcrumbs) && project) {
    if (!breadcrumbsProp?.length) {
      breadcrumbs.unshift({
        bold: !appendBreadcrumbs,
        label: () => title,
      });
    }
  }

  const showMainNavTabs = navigationItems?.length !== 0;
  const layoutTopOffset = HEADER_HEIGHT + (showMainNavTabs ? MAIN_NAV_TAB_BAR_LAYOUT_HEIGHT : 0);
  const tripleLayoutHeaderOffset =
    (headerOffset ?? 0) + (showMainNavTabs ? MAIN_NAV_TAB_BAR_LAYOUT_HEIGHT : 0);

  return (
    <>
      <Head title={title} />

      <Header
        breadcrumbs={breadcrumbs}
        // excludeProject={!addProjectBreadcrumbToCustomBreadcrumbs}
        menuItems={headerMenuItems}
      />

      {showMainNavTabs && (
        <HorizontalMainNavigation
          navigationItems={navigationItems}
        />
      )}

      <ContainerStyle ref={ref} $withMainNavTabs={showMainNavTabs}>
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
            before={before}
            beforeHeader={beforeHeader}
            beforeHeightOffset={layoutTopOffset}
            beforeMousedownActive={mousedownActiveBefore}
            beforeWidth={before ? widthBefore : 0}
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
            setBeforeWidth={setWidthBefore}
          >
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
