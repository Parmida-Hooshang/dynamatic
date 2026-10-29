library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tbf (one-slot transparent buffer, dataless), used as an
-- antitoken store
entity tbf_dataless is
  port (
    clk, rst   : in  std_logic;
    -- input channel
    ins_valid  : in  std_logic;
    ins_ready  : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic;
    -- slot occupancy
    is_full    : out std_logic
  );
end entity;

-- Architecture of tbf
architecture arch of tbf_dataless is
  signal full : std_logic;
begin

  process (clk)
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        full <= '0';
      else
        full <= (not outs_ready and ins_valid) or (full and ins_valid) or (full and not outs_ready);
      end if;
    end if;
  end process;

  outs_valid <= full or ins_valid;
  ins_ready  <= not full or outs_ready;
  is_full    <= full;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of selector
entity selector is
  generic (
    DATA_TYPE : integer
  );
  port (
    -- inputs
    clk, rst         : in std_logic;
    condition        : in std_logic_vector(0 downto 0);
    condition_valid  : in std_logic;
    trueValue        : in std_logic_vector(DATA_TYPE - 1 downto 0);
    trueValue_valid  : in std_logic;
    falseValue       : in std_logic_vector(DATA_TYPE - 1 downto 0);
    falseValue_valid : in std_logic;
    result_ready     : in std_logic;
    -- outputs
    result           : out std_logic_vector(DATA_TYPE - 1 downto 0);
    result_valid     : out std_logic;
    condition_ready  : out std_logic;
    trueValue_ready  : out std_logic;
    falseValue_ready : out std_logic
  );
end entity;

-- Architecture of selector
-- Each data input has a one-slot antitoken store (a tbf). An antitoken is
-- pushed when the select fires without that input's token, and popped when
-- the discarded token arrives. Since the tbf can pop and push in the same
-- cycle, the select can fire again in the cycle the previous discarded token
-- arrives, instead of stalling one cycle to clear the antitoken.
architecture arch of selector is
  signal ee, validInternal     : std_logic;
  signal kill0, kill1          : std_logic;
  signal full0, full1          : std_logic;
  signal atReady0, atReady1    : std_logic;
  signal antitokenStop         : std_logic;
  signal g0, g1                : std_logic;
  signal trueValid, falseValid : std_logic;
begin

  -- While an antitoken is pending on an input, the token arriving there is the
  -- discarded one: it is killed and must not be used.
  trueValid  <= trueValue_valid and not full0;
  falseValid <= falseValue_valid and not full1;

  ee            <= condition_valid and ((not condition(0) and falseValid) or (condition(0) and trueValid)); --condition(0) and one input
  -- Stop if firing would need to push an antitoken into a full store that is
  -- not popped this cycle
  antitokenStop <= not atReady0 or not atReady1;
  validInternal <= ee and not antitokenStop;

  g0 <= not trueValid and validInternal and result_ready;
  g1 <= not falseValid and validInternal and result_ready;

  result_valid     <= validInternal;
  trueValue_ready  <= (not trueValue_valid) or (validInternal and result_ready) or kill0; -- normal join or antitoken
  falseValue_ready <= (not falseValue_valid) or (validInternal and result_ready) or kill1; --normal join or antitoken
  condition_ready  <= (not condition_valid) or (validInternal and result_ready); --like normal join

  result <= falseValue when (condition(0) = '0') else
            trueValue;

  antitoken0 : entity work.tbf_dataless
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => g0,
      ins_ready  => atReady0,
      outs_valid => kill0,
      outs_ready => trueValue_valid,
      is_full    => full0
    );

  antitoken1 : entity work.tbf_dataless
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => g1,
      ins_ready  => atReady1,
      outs_valid => kill1,
      outs_ready => falseValue_valid,
      is_full    => full1
    );

end architecture;
