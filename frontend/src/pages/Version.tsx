import { useRecoilValue } from 'recoil';

import Page from 'pages/Page';

import Box from '@mui/material/Box';

import WelcomeScreen from 'components/organisms/chat/Messages/welcomeScreen';

import { projectSettingsState } from 'state/project';

export default function Version() {
  const projectSettings = useRecoilValue(projectSettingsState);

  return (
    <Page>
      <Box my={2} display="flex" flexGrow={1}>
        <WelcomeScreen
          variant="app"
          markdown={projectSettings?.version}
          allowHtml={projectSettings?.features?.unsafe_allow_html}
          latex={projectSettings?.features?.latex}
        />
      </Box>
    </Page>
  );
}
