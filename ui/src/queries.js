
import gql from "graphql-tag";

const GET_RECENT_AGENTS = gql`
{
  recentAgents {
    ident
    created
    specId
    lastCheckpoint
    tags{
        edges{
            node{
                id
                tagId
            }
        }
    }
    recentSessions {
      ident
      status
      envSpecId
      sessionType
      parentSessionId
      
    }
  }
}
`;

const GET_RECENT_AGENT_SMALL = gql`
{
  recentAgents {
    ident
    created
    specId
    spec{
        id
        ident
        displayedName
    }
  }
}
`;



const GET_ALL_SESSIONS = gql `
  query GetSessions($first: Int, $after: String, $before:String, $archived:Boolean) {
    sessions: allSessions(
      sort: CREATED_DESC
      first: $first
      before: $before
      after: $after,
      filters: {
            archived:$archived  
        }
    ) {
      pageInfo {
        startCursor
        endCursor
        hasNextPage
        hasPreviousPage
      }
      edges {
        node {
            id
          ident
          created
          envSpecId
 
          
          label
          comments
          status
          config
          sessionType
          agentId
            task {
                id
            ident
        }
        
          summary 
        }
      }
    }
  }
`;




export { GET_RECENT_AGENTS,GET_RECENT_AGENT_SMALL,GET_ALL_SESSIONS };
