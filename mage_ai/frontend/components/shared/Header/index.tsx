import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/router';

import AuthToken from '@api/utils/AuthToken';
import Breadcrumbs, { BreadcrumbType as BreadcrumbTypeOrig } from '@components/Breadcrumbs';
import Button from '@oracle/elements/Button';
import ClickOutside from '@oracle/components/ClickOutside';
import ClientOnly from '@hocs/ClientOnly';
import Flex from '@oracle/components/Flex';
import FlexContainer from '@oracle/components/FlexContainer';
import FlyoutMenu from '@oracle/components/FlyoutMenu';
import GitActions from '@components/VersionControl/GitActions';
import KeyboardShortcutButton from '@oracle/elements/Button/KeyboardShortcutButton';
import LaunchKeyboardShortcutText from '@components/CommandCenter/LaunchKeyboardShortcutText';
import Loading, { LoadingStyleEnum } from '@oracle/components/Loading';
import PopupMenu from '@oracle/components/PopupMenu';
import ProjectType, { FeatureUUIDEnum } from '@interfaces/ProjectType';
import Spacing from '@oracle/elements/Spacing';
import Text from '@oracle/elements/Text';
import api from '@api';
import useDelayFetch from '@api/utils/useDelayFetch';
import useProject from '@utils/models/project/useProject';
import { YELLOW } from '@oracle/styles/colors/main';
import { BranchAlt, MageProLetters, Planet, Slack, UFO } from '@oracle/icons';
import { HeaderStyle } from './index.style';
import { CommandCenterStateEnum } from '@interfaces/CommandCenterType';
import { CustomEventUUID, CUSTOM_EVENT_NAME_COMMAND_CENTER_STATE_CHANGED } from '@utils/events/constants';
import { LinkStyle } from '@components/PipelineDetail/FileHeaderMenu/index.style';
import { REQUIRE_USER_AUTHENTICATION } from '@utils/session';
import { PADDING_UNITS, UNIT } from '@oracle/styles/units/spacing';
import { getSetSettings } from '@storage/CommandCenter/utils';
import { launchCommandCenter } from '@components/CommandCenter/utils';
import { pauseEvent } from '@utils/events';
import { storeLocalTimezoneSetting } from '@components/settings/workspace/utils';
import { useModal } from '@context/Modal';
import { useError } from '@context/Error';

export type BreadcrumbType = BreadcrumbTypeOrig;

export type MenuItemType = {
  label: () => string;
  onClick: () => void;
  openConfirmationDialogue?: boolean;
  uuid: string;
};

export type HeaderProps = {
  breadcrumbs?: BreadcrumbType[];
  hideActions?: boolean;
  menuItems?: MenuItemType[];
  project?: ProjectType;
  version?: string;
};

function Header({
  breadcrumbs: breadcrumbsProp,
  hideActions,
  menuItems,
  project: projectProp,
  version: versionProp,
}: HeaderProps) {
  const [showError] = useError(null, {}, [], {
    uuid: 'shared/Header',
  });

  const router = useRouter();

  const [commandCenterState, setCommandCenterState] = useState<CommandCenterStateEnum>(null);
  const [enableCommandCenterLoading, setEnableCommandCenterLoading] = useState<boolean>(false);
  const [highlightedMenuIndex, setHighlightedMenuIndex] = useState<number>(null);
  const [confirmationDialogueOpen, setConfirmationDialogueOpen] = useState<boolean>(false);
  const [confirmationAction, setConfirmationAction] = useState(null);

  const menuRef = useRef(null);
  const projectRef = useRef(null);

  const loggedIn = AuthToken.isLoggedIn();
  const {
    data: dataGitBranch,
    mutate: fetchBranch,
  } = useDelayFetch(api.git_branches.detail,
    'test',
    {
      _format: 'with_basic_details',
    },
    {
      revalidateOnFocus: false,
    }, {
    pauseFetch: REQUIRE_USER_AUTHENTICATION() && !loggedIn,
  },
    {
      delay: 11000,
    },
  );
  const {
    is_git_integration_enabled: gitIntegrationEnabled,
    name: branch,
  } = useMemo(() => dataGitBranch?.['git_branch'] || {}, [dataGitBranch]);

  const {
    featureEnabled,
    featureUUIDs,
    isLoadingProject,
    isLoadingUpdate,
    project: projectInit,
    rootProject,
    updateProject,
  } = useProject({ showError: hideActions ? null : showError });
  const project = useMemo(() => projectProp || projectInit, [projectInit, projectProp]);
  const version = useMemo(() => versionProp || project?.version, [project, versionProp]);
  const commandCenterEnabled = useMemo(() =>
    CommandCenterStateEnum.CLOSED === commandCenterState
    || CommandCenterStateEnum.OPEN === commandCenterState
    || featureEnabled?.(featureUUIDs?.COMMAND_CENTER), [
    commandCenterState,
    featureEnabled,
    featureUUIDs,
  ]);
  projectRef.current = project;

  const launchCommandCenterWrapper = useCallback(() => {
    if (commandCenterEnabled) {
      launchCommandCenter();
    } else {
      setEnableCommandCenterLoading(true);
      updateProject({
        features: {
          ...(project?.features || {}),
          [featureUUIDs?.COMMAND_CENTER]: true,
        },
      }).then((response) => {
        if (response?.data?.error) {
          setEnableCommandCenterLoading(false);
          showError({
            errors: response?.data?.error,
            response,
          });
        } else {
          if (typeof window !== 'undefined') {
            const eventCustom = new CustomEvent(CustomEventUUID.COMMAND_CENTER_ENABLED);
            window.dispatchEvent(eventCustom);
          }
        }
      });
    }
  }, [
    commandCenterEnabled,
    featureUUIDs?.COMMAND_CENTER,
    project?.features,
    showError,
    updateProject,
  ]);

  const breadcrumbProjects = [];
  if (rootProject) {
    breadcrumbProjects.push({
      label: () => rootProject?.name,
      linkProps: {
        href: '/',
      },
    });
  }

  if (project) {
    const crumb: BreadcrumbType = {
      label: () => project?.name,
    };

    if (rootProject) {
      crumb.loading = isLoadingUpdate && !enableCommandCenterLoading;
      crumb.options = Object.keys(rootProject?.projects || {}).map((projectName: string) => ({
        onClick: () => {
          updateProject({
            activate_project: projectName,
          }).then((response) => {
            if (response?.data?.error) {
              showError({
                errors: response?.data?.error,
                response,
              });
            } else {
              const displayLocalTimeUpdated: boolean = !!response?.data?.project?.features?.display_local_timezone;
              storeLocalTimezoneSetting(displayLocalTimeUpdated);
              if (typeof window !== 'undefined') {
                window.location.reload();
              }
            }
          });
        },
        selected: projectName === project?.name,
        uuid: projectName,
      }));
    } else {
      crumb.linkProps = {
        href: '/',
      };
    }

    breadcrumbProjects.push(crumb);
  } else if (!isLoadingProject && !hideActions) {
    breadcrumbProjects.push({
      bold: true,
      danger: true,
      label: () => 'Error loading project configuration',
    });
  }

  const breadcrumbs = useMemo(() => [
    ...breadcrumbProjects,
    ...(breadcrumbsProp || []),
  ], [
    breadcrumbProjects,
    breadcrumbsProp,
    project,
  ]);
  const { pipeline: pipelineUUID } = router.query;

  const { latest_version: latestVersion } = project || {};

  const [showModal, hideModal] = useModal(() => (
    <GitActions
      branch={branch}
      fetchBranch={fetchBranch}
    />
  ), {}, [branch, fetchBranch], {
    background: true,
    uuid: 'git_actions',
  });

  const branchName = useMemo(() => {
    if (branch?.length >= 21) {
      return `${branch.slice(0, 21)}...`;
    }

    return branch;
  }, [branch]);

  useEffect(() => {
    const handleState = ({
      detail,
    }) => {
      if (detail?.state) {
        setCommandCenterState(detail?.state);

        if (CommandCenterStateEnum.MOUNTED === detail?.state) {
          // Only launch this if it was previously disabled.
          // The feature can be enabled by clicking the button in the header.
          if (!projectRef?.current?.features?.[FeatureUUIDEnum.COMMAND_CENTER]) {
            setTimeout(() => {
              launchCommandCenter();
              setEnableCommandCenterLoading(false);
            }, 1);
          }
        }
      }
    };

    if (typeof window !== 'undefined') {
      // @ts-ignore
      window.addEventListener(CUSTOM_EVENT_NAME_COMMAND_CENTER_STATE_CHANGED, handleState);
    }

    return () => {
      if (typeof window !== 'undefined') {
        // @ts-ignore
        window.removeEventListener(CUSTOM_EVENT_NAME_COMMAND_CENTER_STATE_CHANGED, handleState);
      }
    };
  }, []);

  return (
    <HeaderStyle>
      <ClientOnly>
        <FlexContainer
          alignItems="center"
          fullHeight
          justifyContent="space-between"
        >
          <Flex alignItems="center">
            <Breadcrumbs
              breadcrumbs={breadcrumbs}
            />
          </Flex>

          <Flex alignItems="center">
            {gitIntegrationEnabled && branch && (
              <Spacing mr={1}>
                <KeyboardShortcutButton
                  compact
                  highlightOnHoverAlt
                  noBackground
                  noHoverUnderline
                  onClick={showModal}
                  sameColorAsText
                  title={branch}
                  uuid="Header/GitActions"
                >
                  <FlexContainer alignItems="center">
                    <BranchAlt size={1.5 * UNIT} />
                    <Spacing ml={1} />
                    <Text monospace noWrapping small>
                      {branchName}
                    </Text>
                  </FlexContainer>
                </KeyboardShortcutButton>
              </Spacing>
            )}

            {latestVersion && version && latestVersion !== version && (
              <Button
                backgroundColor={YELLOW}
                borderLess
                compact
                linkProps={{
                  href: 'https://docs.mage.ai/about/releases',
                }}
                noHoverUnderline
                pill
                sameColorAsText
                target="_blank"
                title={`Update to version ${latestVersion}`}
              >
                <Text black bold>Update</Text>
              </Button>
            )}

            {/* <Spacing ml={1}>
              <KeyboardShortcutButton
                beforeElement={<Slack />}
                compact
                highlightOnHoverAlt
                inline
                linkProps={{
                  as: 'https://www.mage.ai/chat',
                  href: 'https://www.mage.ai/chat',
                }}
                noBackground
                noHoverUnderline
                openNewTab
                sameColorAsText
                uuid="Header/live_chat"
              >
                Live help
              </KeyboardShortcutButton>
            </Spacing>

            <Spacing ml={1}>
              <KeyboardShortcutButton
                compact
                highlightOnHoverAlt
                inline
                linkProps={{
                  as: 'https://cloud.mage.ai/sign-up?ref=oss',
                  href: 'https://cloud.mage.ai/sign-up?ref=oss',
                }}
                openNewTab
                noBackground
                noHoverUnderline
                sameColorAsText
                afterElement={<MageProLetters size={24} />}
                uuid="Header/pro"
              >
                Try
              </KeyboardShortcutButton>
            </Spacing>*/}

            {menuItems &&
              <>
                <Spacing ml={2} />

                <ClickOutside
                  onClickOutside={() => setHighlightedMenuIndex(null)}
                  open
                  style={{
                    position: 'relative',
                  }}
                >
                  <FlexContainer>
                    <LinkStyle
                      highlighted={highlightedMenuIndex === 0}
                      onClick={() => setHighlightedMenuIndex(val => val === 0 ? null : 0)}
                      onMouseEnter={() => setHighlightedMenuIndex(val => val !== null ? 0 : null)}
                      ref={menuRef}
                    >
                      <Text>
                        Menu
                      </Text>
                    </LinkStyle>

                    <FlyoutMenu
                      alternateBackground
                      items={menuItems}
                      onClickCallback={() => setHighlightedMenuIndex(null)}
                      open={highlightedMenuIndex === 0}
                      parentRef={menuRef}
                      rightOffset={0}
                      setConfirmationAction={setConfirmationAction}
                      setConfirmationDialogueOpen={setConfirmationDialogueOpen}
                      uuid="PipelineDetail/Header/menu"
                    />
                  </FlexContainer>
                </ClickOutside>

                <ClickOutside
                  onClickOutside={() => setConfirmationDialogueOpen(false)}
                  open={confirmationDialogueOpen}
                >
                  <PopupMenu
                    danger
                    onCancel={() => setConfirmationDialogueOpen(false)}
                    onClick={confirmationAction}
                    right={UNIT * 16}
                    subtitle="This is irreversible and will immediately delete everything associated with the pipeline, including its blocks, triggers, runs, logs, and history."
                    title={`Are you sure you want to delete the pipeline ${pipelineUUID}?`}
                    width={UNIT * 40}
                  />
                </ClickOutside>
              </>
            }
          </Flex>
        </FlexContainer>
      </ClientOnly>
    </HeaderStyle>
  );
}

export default Header;
