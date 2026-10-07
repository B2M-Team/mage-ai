import NextLink from 'next/link';

import FlexContainer from '@oracle/components/FlexContainer';
import Link from '@oracle/elements/Link';
import Spacing from '@oracle/elements/Spacing';
import { ItemStyle, SectionTitleStyle } from './index.style';
import { PADDING_VERTICAL_UNITS, UNIT } from '@oracle/styles/units/spacing';

export type SectionItemType = {
  Icon?: any;
  label?: () => string;
  linkProps?: {
    as?: string;
    href: string;
  };
  onClick?: (e: any) => void;
  uuid: string;
};

export type SectionLinksType = {
  items: SectionItemType[];
  title?: () => string;
  uuid: string;
};

type VerticalSectionLinksProps = {
  isItemSelected?: (item: SectionItemType & {
    uuidWorkspace: string;
  }) => boolean;
  sections: SectionLinksType[];
};

function VerticalSectionLinks({
  isItemSelected,
  sections,
}: VerticalSectionLinksProps) {
  return (
    <Spacing py={PADDING_VERTICAL_UNITS}>
      {sections?.map(({
        items,
        title,
        uuid,
      }) => (
        <Spacing key={uuid}>
          <SectionTitleStyle>
            {title ? title() : uuid}
          </SectionTitleStyle>

          {items?.map((item) => {
            const {
              Icon: IconItem,
              label,
              linkProps,
              onClick,
              uuid: uuidItem,
            } = item;
            const selected = isItemSelected?.({
              ...item,
              uuidWorkspace: uuid,
            });

            const linkLabel = label ? label() : uuidItem;
            const el = (
              <ItemStyle
                selected={selected}
              >
                <FlexContainer alignItems="center">
                  {IconItem && (
                    <>
                      <IconItem fill="#3F3F46" size={20} />

                      <Spacing mr={1} />
                    </>
                  )}

                  {linkLabel}
                </FlexContainer>
              </ItemStyle>
            );

            if (linkProps) {
              return (
                <NextLink
                  {...linkProps}
                  key={uuidItem}
                  passHref
                >
                  <Link
                    block
                    color="#3F3F46"
                    noHoverUnderline
                    noOutline
                  >
                    {el}
                  </Link>
                </NextLink>
              );
            }

            return (
              <Link
                block
                color="#3F3F46"
                key={uuidItem}
                noHoverUnderline
                noOutline
                onClick={onClick}
                preventDefault
              >
                {el}
              </Link>
            );
          })}
        </Spacing>
      ))}
    </Spacing>
  );
}

export default VerticalSectionLinks;
