import { BoxProps } from '@chakra-ui/react';

// 22px --> 40px 字体调整后
export const textareaMinH = '47px';

export const MessageCardStyle: BoxProps = {
  px: 4,
  py: 3,
  borderRadius: '0 8px 8px 8px',
  boxShadow: 'none',
  display: 'inline-block',
  maxW: ['calc(100% - 25px)', 'calc(100% - 40px)'],
  // color: 'myGray.900'
};

export enum FeedbackTypeEnum {
  user = 'user',
  admin = 'admin',
  hidden = 'hidden'
}
