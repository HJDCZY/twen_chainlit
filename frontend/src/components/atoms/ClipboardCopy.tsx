import { useState } from 'react';
import { useCopyToClipboard } from 'usehooks-ts';

import ContentPaste from '@mui/icons-material/ContentPaste';
import IconButton, { IconButtonProps } from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';

interface ClipboardCopyProps {
  value: string;
  theme?: 'dark' | 'light';
  edge?: IconButtonProps['edge'];
}

const ClipboardCopy: React.FC<ClipboardCopyProps> = ({ value, edge }) => {
  const [isCopied, setIsCopied] = useState(false);

  // 使用传统的document.execCommand方法作为回退
  const handleCopy = () => {
    // 创建一个临时的输入元素
    const tempInput = document.createElement('input');
    document.body.appendChild(tempInput);
    tempInput.value = value;
    tempInput.select();
    try {
      // 尝试执行复制命令
      const successful = document.execCommand('copy');
      if (successful) {
        setIsCopied(true);
      } else {
        throw new Error('Copy command failed');
      }
    } catch (err) {
      console.error('Failed to copy text: ', err);
    } finally {
      // 清理：移除临时输入元素
      document.body.removeChild(tempInput);
    }
  };

  const handleTooltipClose = () => {
    setIsCopied(false);
  };

  return (
    <Tooltip
      title={isCopied ? 'Copied to clipboard!' : 'Copy'}
      onClose={handleTooltipClose}
      sx={{ zIndex: 2 }}
    >
      <IconButton color="inherit" edge={edge} onClick={handleCopy}>
        <ContentPaste sx={{ height: 16, width: 16 }} />
      </IconButton>
    </Tooltip>
  );
};

// const ClipboardCopy = ({ value, edge }: ClipboardCopyProps): JSX.Element => {
//   const [isCopied, setIsCopied] = useState(false);
//   const [_, copy] = useCopyToClipboard();
//
//   const handleCopy = () => {
//     copy(value)
//       .then(() => {
//         setIsCopied(true);
//       })
//       .catch((err) => console.log('An error occurred while copying: ', err));
//   };
//
//   const handleTooltipClose = () => {
//     setIsCopied(false);
//   };
//
//   return (
//     <Tooltip
//       title={isCopied ? '复制成功' : '复制'}
//       onClose={handleTooltipClose}
//       sx={{ zIndex: 2 }}
//     >
//       <IconButton color="inherit" edge={edge} onClick={handleCopy}>
//         <ContentPaste sx={{ height: 16, width: 16 }} />
//       </IconButton>
//     </Tooltip>
//   );
// };

export { ClipboardCopy };
