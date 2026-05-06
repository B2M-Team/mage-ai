import FlyoutMenuWrapper from '@oracle/components/FlyoutMenu/FlyoutMenuWrapper';
import KeyboardShortcutButton from '@oracle/elements/Button/KeyboardShortcutButton';
import { Add } from '@oracle/icons';
import { BUTTON_GRADIENT } from '@oracle/styles/colors/gradients';
import { FlyoutMenuItemType } from '@oracle/components/FlyoutMenu';
import { UNIT } from '@oracle/styles/units/spacing';

export const SHARED_BUTTON_PROPS = {
  bold: true,
  greyBorder: true,
  paddingBottom: 9,
  paddingTop: 9,
};

export type AddButtonProps = {
  addButtonMenuOpen: boolean;
  addButtonMenuRef?: React.RefObject<any>;
  /** Solid fill replaces gradient; label and icon use white. */
  backgroundColor?: string;
  isLoading?: boolean;
  label: string;
  menuItems: FlyoutMenuItemType[];
  onClick: () => void;
  onClickCallback: () => void;
};

function AddButton({
  addButtonMenuOpen,
  addButtonMenuRef,
  backgroundColor,
  isLoading,
  label,
  menuItems,
  onClick,
  onClickCallback,
}: AddButtonProps) {
  const solid = !!backgroundColor;

  return (
    <FlyoutMenuWrapper
      disableKeyboardShortcuts
      items={menuItems}
      onClickCallback={onClickCallback}
      onClickOutside={onClickCallback}
      open={addButtonMenuOpen}
      parentRef={addButtonMenuRef}
      roundedStyle
      topOffset={1}
      uuid="Table/Toolbar/NewItemMenu"
    >
      <KeyboardShortcutButton
        {...SHARED_BUTTON_PROPS}
        {...(solid
          ? {
            backgroundColor,
            greyBorder: false,
            noHover: true,
            style: {
              color: '#fff',
            },
          }
          : {
            background: BUTTON_GRADIENT,
          })}
        beforeElement={
          <Add
            size={2.5 * UNIT}
            {...(solid ? { fill: '#ffffff' } : {})}
          />
        }
        loading={isLoading}
        onClick={e => {
          e.preventDefault();
          onClick?.();
        }}
        uuid="shared/AddButton/index"
      >
        {label}
      </KeyboardShortcutButton>
    </FlyoutMenuWrapper>
  );
}

export default AddButton;

