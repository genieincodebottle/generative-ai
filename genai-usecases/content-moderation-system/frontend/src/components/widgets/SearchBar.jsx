import React from 'react';
import { Paper, TextField, Stack, InputAdornment, IconButton } from '@mui/material';
import { Search as SearchIcon, FilterList as FilterIcon } from '@mui/icons-material';

const SearchBar = ({ value, onChange, onFilterClick, placeholder = 'Search...' }) => {
  return (
    <Paper elevation={1} sx={{ p: 2 }}>
      <Stack direction="row" spacing={2} alignItems="center">
        <TextField
          fullWidth
          placeholder={placeholder}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
          sx={{ flex: 1 }}
        />
        {onFilterClick && (
          <IconButton onClick={onFilterClick}>
            <FilterIcon />
          </IconButton>
        )}
      </Stack>
    </Paper>
  );
};

export default SearchBar;
