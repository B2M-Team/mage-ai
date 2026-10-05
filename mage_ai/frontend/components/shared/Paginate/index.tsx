import React from 'react';
import styled from 'styled-components';

import FlexContainer from '@oracle/components/FlexContainer';
import Spacing from '@oracle/elements/Spacing';
import { PaginateArrowLeft, PaginateArrowRight } from '@oracle/icons';
import { UNIT } from '@oracle/styles/units/spacing';

type PaginateProps = {
  page: number;
  maxPages: number;
  onUpdate: (page: number) => void;
  totalPages: number;
};

export const ROW_LIMIT = 30;
export const MAX_PAGES = 9;

const PAGE_TEXT = '#18181B';
const PAGE_BORDER = '#E4E4E7';
const PAGE_HOVER = '#F7F7F7';
const PAGE_DISABLED = '#D1D5DB';

const PageButtonStyle = styled.button<{
  $active?: boolean;
}>`
  align-items: center;
  background-color: ${props => props.$active ? '#F3F4F6' : '#FFFFFF'};
  border: 1px solid ${props => props.$active ? '#18181B' : PAGE_BORDER};
  border-radius: 6px;
  color: ${PAGE_TEXT};
  cursor: pointer;
  display: inline-flex;
  font-family: inherit;
  font-size: 14px;
  font-weight: ${props => props.$active ? 500 : 400};
  height: 40px;
  justify-content: center;
  line-height: 20px;
  min-width: 40px;
  padding: 0 ${UNIT}px;

  &:hover:not(:disabled) {
    background-color: ${PAGE_HOVER};
  }

  &:disabled {
    background-color: #FFFFFF;
    border-color: ${PAGE_BORDER};
    color: ${PAGE_DISABLED};
    cursor: not-allowed;
  }
`;

function Paginate({
  page,
  maxPages: maxPagesProp,
  onUpdate,
  totalPages,
}: PaginateProps) {

  let pageArray = [];

  let maxPages = maxPagesProp;
  if (maxPages > totalPages) {
    pageArray = Array.from({ length: totalPages }, (_, i) => i);
  } else {
    const interval = Math.floor(maxPages / 2);
    let minPage = page - interval;
    if (page + interval >= totalPages) {
      minPage = totalPages - maxPages + 2;
      maxPages -= 2;
    } else if (page - interval <= 0) {
      minPage = 0;
      maxPages -= 2;
    } else {
      maxPages -= 4;
      minPage = page - Math.floor(maxPages / 2);
    }
  
    pageArray = Array.from({ length: maxPages }, (_, i) => i + minPage);
  }

  const iconStroke = (disabled: boolean) => disabled ? PAGE_DISABLED : PAGE_TEXT;

  return (
    <>
      {totalPages > 0 && (
        <FlexContainer alignItems="center">
          <PageButtonStyle
            disabled={page === 0}
            onClick={() => onUpdate(page - 1)}
            type="button"
          >
            <PaginateArrowLeft size={1.5 * UNIT} stroke={iconStroke(page === 0)} />
          </PageButtonStyle>
          {!pageArray.includes(0) && (
            <>
              <Spacing key={0} ml={1}>
                <PageButtonStyle
                  onClick={() => onUpdate(0)}
                  type="button"
                >
                  {1}
                </PageButtonStyle>
              </Spacing>
              {!pageArray.includes(1) && (
                <Spacing key={0} ml={1}>
                  <PageButtonStyle
                    disabled
                    type="button"
                  >
                    ...
                  </PageButtonStyle>
                </Spacing>
              )}
            </>
          )}
          {pageArray.map((p) => (
            <Spacing key={p} ml={1}>
              <PageButtonStyle
                $active={p === page}
                onClick={() => {
                  if (p !== page) {
                    onUpdate(p);
                  }
                }}
                type="button"
              >
                {p + 1}
              </PageButtonStyle>
            </Spacing>
          ))}
          {!pageArray.includes(totalPages - 1) && (
            <>
              {!pageArray.includes(totalPages - 2) && (
                <Spacing key={0} ml={1}>
                  <PageButtonStyle
                    disabled
                    type="button"
                  >
                    ...
                  </PageButtonStyle>
                </Spacing>
              )}
              <Spacing key={totalPages - 1} ml={1}>
                <PageButtonStyle
                  onClick={() => onUpdate(totalPages - 1)}
                  type="button"
                >
                  {totalPages}
                </PageButtonStyle>
              </Spacing>
            </>
          )}
          <Spacing ml={1} />
          <PageButtonStyle
            disabled={page === totalPages - 1}
            onClick={() => onUpdate(page + 1)}
            type="button"
          >
            <PaginateArrowRight size={1.5 * UNIT} stroke={iconStroke(page === totalPages - 1)} />
          </PageButtonStyle>
        </FlexContainer>
      )}
    </>
  );
}

export default Paginate;
