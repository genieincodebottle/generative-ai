import React from 'react';
import { Box, Container, Typography, Link } from '@mui/material';
import { styled } from '@mui/system';
import GitHubIcon from '@mui/icons-material/GitHub';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import TwitterIcon from '@mui/icons-material/Twitter';
import InstagramIcon from '@mui/icons-material/Instagram';

const FooterContainer = styled(Box)(({ theme }) => ({
  backgroundColor: theme.palette.primary.main,
  color: theme.palette.common.white,
  padding: theme.spacing(2),
  marginTop: 'auto',
}));

const SocialIcons = styled(Box)({
  display: 'flex',
  justifyContent: 'center',
  gap: '1rem',
  marginTop: '0.5rem',
});

const SocialIcon = styled(Link)(({ theme }) => ({
  color: theme.palette.common.white,
  fontSize: '1.5rem',
  transition: 'color 0.3s ease',
  '&:hover': {
    color: theme.palette.secondary.main,
  },
}));

const Footer: React.FC = () => {
  return (
    <FooterContainer component="footer">
      <Container maxWidth="lg">
        <Typography variant="body2" align="center">
          &copy; 2024 @genieincodebottle
        </Typography>
        <SocialIcons>
          <SocialIcon href="https://github.com/genieincodebottle/generative-ai" target="_blank" rel="noopener noreferrer">
            <GitHubIcon />
          </SocialIcon>
          <SocialIcon href="https://linkedin.com/in/rajesh-srivastava" target="_blank" rel="noopener noreferrer">
            <LinkedInIcon />
          </SocialIcon>
          <SocialIcon href="https://www.instagram.com/genieincodebottle/" target="_blank" rel="noopener noreferrer">
            <InstagramIcon />
          </SocialIcon>
          <SocialIcon href="https://x.com/genie_aicode" target="_blank" rel="noopener noreferrer">
            <TwitterIcon />
          </SocialIcon>
        </SocialIcons>
      </Container>
    </FooterContainer>
  );
};

export default Footer;